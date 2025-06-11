use std::{
    ffi,
    mem::{self, offset_of},
    ops::Range,
    ptr,
    time::Instant,
};

use ash::vk::{self, PipelineMultisampleStateCreateInfo};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::BgColor;

use super::Rect;

#[derive(Debug, Clone)]
#[repr(C)]
struct GpuRect {
    pos: [u32; 2],
    size: [u32; 2],
    bg_color: BgColor,
    parent_idx: u32,
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

pub unsafe fn unsafe_main(rectangles: &[Rect], width: u32, height: u32) {
    fn push_rectangles(gpu_rects: &mut Vec<GpuRect>, rects: &[Rect], parent_idx: u32) {
        for rect in rects {
            let idx = gpu_rects.len() as u32 + 1;

            gpu_rects.push(GpuRect {
                bg_color: rect.bg_color,
                pos: rect.pos,
                size: rect.size,
                parent_idx,
            });

            push_rectangles(gpu_rects, &rect.children, idx);
        }
    }

    let mut gpu_rects = vec![GpuRect {
        bg_color: BgColor([1.0, 1.0, 1.0, 1.0]),
        pos: [0, 0],
        size: [width, height],
        parent_idx: 0,
    }];
    push_rectangles(&mut gpu_rects, rectangles, 1);
    let mut rectangles = gpu_rects;
    println!("{:?}", rectangles.len());

    let entry = ash::Entry::load().unwrap();

    let instance = {
        entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3))
                .enabled_layer_names(&["VK_LAYER_KHRONOS_validation".as_ptr() as *const i8]),
            None,
        )
    }
    .unwrap();

    let mut physical_devices = {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
    };

    physical_devices.sort_by_key(|&device| {
        match instance.get_physical_device_properties(device).device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 3,
            vk::PhysicalDeviceType::OTHER => 4,
            _ => 5,
        }
    });

    let physical_device = physical_devices
        .get(0)
        .copied()
        .expect("No physical devices found");

    eprintln!(
        "physical device name: {}",
        instance
            .get_physical_device_properties(physical_device)
            .device_name_as_c_str()
            .unwrap()
            .to_str()
            .unwrap()
    );

    let queue_family_idx = instance
        .get_physical_device_queue_family_properties(physical_device)
        .into_iter()
        .enumerate()
        .filter_map(|(i, props)| {
            if props
                .queue_flags
                .contains(vk::QueueFlags::COMPUTE & vk::QueueFlags::GRAPHICS)
            {
                Some(i)
            } else {
                None
            }
        })
        .next()
        .expect("No queue family found") as u32;

    let device = {
        instance
            .create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .enabled_features(&vk::PhysicalDeviceFeatures::default().dual_src_blend(true))
                    .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family_idx)
                        .queue_priorities(&[1.0])])
                    .push_next(
                        &mut vk::PhysicalDeviceDynamicRenderingFeatures::default()
                            .dynamic_rendering(true),
                    )
                    .enabled_extension_names(&[vk::KHR_DYNAMIC_RENDERING_NAME.as_ptr()]),
                None,
            )
            .unwrap()
    };

    let command_pool = {
        device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_idx),
                None,
            )
            .unwrap()
    };

    let memory_properties = instance.get_physical_device_memory_properties(physical_device);

    let (image, view) = {
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .extent(vk::Extent3D {
                        depth: 1,
                        height,
                        width,
                    })
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .image_type(vk::ImageType::TYPE_2D)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .mip_levels(1)
                    .array_layers(1),
                None,
            )
            .unwrap();

        let requirements = device.get_image_memory_requirements(image);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(
                find_memorytype_index(
                    &requirements,
                    &memory_properties,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .unwrap(),
            );

        let memory = device.allocate_memory(&allocate_info, None).unwrap();
        device.bind_image_memory(image, memory, 0).unwrap();

        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        let view = device.create_image_view(&view_create_info, None).unwrap();
        (image, view)
    };

    let (download_image_buffer, download_image_buffer_memory) = {
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size((width * height * 4) as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
            .unwrap();

        let requirements = device.get_buffer_memory_requirements(buffer);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(
                find_memorytype_index(
                    &requirements,
                    &memory_properties,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .unwrap(),
            );

        let memory = device.allocate_memory(&allocate_info, None).unwrap();
        device.bind_buffer_memory(buffer, memory, 0).unwrap();

        (buffer, memory)
    };

    let visible_vertex_buffer = {
        let size = size_of_val(&*rectangles) as u64;
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
            .unwrap();

        let requirements = device.get_buffer_memory_requirements(buffer);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(
                find_memorytype_index(
                    &requirements,
                    &memory_properties,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .unwrap(),
            );

        let memory = device.allocate_memory(&allocate_info, None).unwrap();
        device.bind_buffer_memory(buffer, memory, 0).unwrap();

        let copy_start = Instant::now();
        // for rect in &mut rectangles {
        //     rect.pos = [4.0, 7.0];
        // }

        let ptr = device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .unwrap();
        ptr::copy_nonoverlapping(rectangles.as_ptr() as *mut _, ptr, size as usize);
        device.unmap_memory(memory);
        eprintln!("copy: {:?}", copy_start.elapsed());

        buffer
    };

    let queue = device.get_device_queue(queue_family_idx, 0);

    let vertex_buffer = {
        let size = size_of_val(&*rectangles) as u64;
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                ),
                None,
            )
            .unwrap();

        let requirements = device.get_buffer_memory_requirements(buffer);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(
                find_memorytype_index(
                    &requirements,
                    &memory_properties,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .unwrap(),
            );

        let memory = device.allocate_memory(&allocate_info, None).unwrap();
        device.bind_buffer_memory(buffer, memory, 0).unwrap();

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffer = device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()[0];

        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

        device.cmd_copy_buffer(
            command_buffer,
            visible_vertex_buffer,
            buffer,
            &[vk::BufferCopy::default().size(size)],
        );

        device.end_command_buffer(command_buffer).unwrap();

        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                vk::Fence::null(),
            )
            .unwrap();

        device.queue_wait_idle(queue).unwrap();

        buffer
    };

    let vertex_shader = {
        device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(bytemuck::cast_slice(include_bytes!("./shader.vert.spv"))),
                None,
            )
            .unwrap()
    };

    let fragment_shader = {
        device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(bytemuck::cast_slice(include_bytes!("./shader.frag.spv"))),
                None,
            )
            .unwrap()
    };

    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)];

    let set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    let set_layout = device
        .create_descriptor_set_layout(&set_layout_create_info, None)
        .unwrap();

    let set_layouts = [set_layout];

    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(&[vk::PushConstantRange::default()
                    .size(mem::size_of::<[f32; 2]>() as u32)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)])
                .set_layouts(&set_layouts),
            None,
        )
        .unwrap();

    let shader_entry_name = ffi::CStr::from_bytes_with_nul_unchecked(b"main\0");
    let shader_stage_create_infos = [
        vk::PipelineShaderStageCreateInfo {
            module: vertex_shader,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            module: fragment_shader,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let vertex_input_binding_descriptions = [];

    let vertex_input_attribute_descriptions = [];

    let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: width as f32,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        extent: vk::Extent2D {
            width: width.div_ceil(1),
            height: height.div_ceil(1),
        },
        offset: vk::Offset2D { x: 0, y: 0 },
    }];
    let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
        .scissors(&scissors)
        .viewports(&viewports);

    let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        ..Default::default()
    };

    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::TRUE,
        src_color_blend_factor: vk::BlendFactor::SRC1_COLOR,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    }];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op(vk::LogicOp::CLEAR)
        .attachments(&color_blend_attachment_states);

    let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
        .vertex_binding_descriptions(&vertex_input_binding_descriptions);

    let multisample_state_info = PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&[vk::Format::R8G8B8A8_UNORM]);

    let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stage_create_infos)
        .vertex_input_state(&vertex_input_state_info)
        .input_assembly_state(&vertex_input_assembly_state_info)
        .viewport_state(&viewport_state_info)
        .rasterization_state(&rasterization_info)
        .color_blend_state(&color_blend_state)
        .multisample_state(&multisample_state_info)
        .layout(pipeline_layout)
        .push_next(&mut pipeline_rendering_create_info);

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[graphic_pipeline_info], None)
        .unwrap()[0];

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
    }];

    let descriptor_pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
        .unwrap();

    let descriptor_set = device
        .allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts),
        )
        .unwrap()[0];

    device.update_descriptor_sets(
        &[vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&[vk::DescriptorBufferInfo::default()
                .buffer(vertex_buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)])],
        &[],
    );

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    for i in 0..5 {
        let command_buffer = device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()[0];

        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        };

        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::COLOR_ATTACHMENT_READ,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .level_count(1),
                )
                .image(image)],
        );

        device.cmd_begin_rendering(
            command_buffer,
            &vk::RenderingInfo::default()
                .render_area(scissors[0])
                .layer_count(1)
                .color_attachments(&[vk::RenderingAttachmentInfo::default()
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(view)
                    .clear_value(clear_value)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)]),
        );

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );

        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytemuck::bytes_of(&[width as f32, height as f32]),
        );

        device.cmd_draw(command_buffer, 6, rectangles.len() as u32, 0, 0);

        device.cmd_end_rendering(command_buffer);

        device.end_command_buffer(command_buffer).unwrap();

        let start = Instant::now();
        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                fence,
            )
            .unwrap();

        device.wait_for_fences(&[fence], true, !0).unwrap();

        let duration = start.elapsed();
        println!("Duration: {:?}", duration);
    }
    let fence = device
        .create_fence(&vk::FenceCreateInfo::default(), None)
        .unwrap();
    let command_buffer = device
        .allocate_command_buffers(&command_buffer_allocate_info)
        .unwrap()[0];

    device
        .begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .unwrap();

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            )
            .image(image)],
    );

    device.cmd_copy_image_to_buffer(
        command_buffer,
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        download_image_buffer,
        &[vk::BufferImageCopy::default()
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )],
    );

    device.end_command_buffer(command_buffer).unwrap();

    device.reset_fences(&[fence]).unwrap();

    device
        .queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        )
        .unwrap();

    device.wait_for_fences(&[fence], true, !0).unwrap();

    let mut image_buffer = vec![0u8; (width * height * 4) as usize];

    {
        let ptr = device
            .map_memory(
                download_image_buffer_memory,
                0,
                (width * height * 4) as u64,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        ptr::copy_nonoverlapping(
            ptr as *mut _,
            image_buffer.as_mut_ptr(),
            (width * height * 4) as usize,
        );
        device.unmap_memory(download_image_buffer_memory);
    }

    image::save_buffer(
        "image.png",
        &image_buffer,
        width,
        height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}
