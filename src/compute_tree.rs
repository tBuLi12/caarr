use std::{ffi, mem, ops::Range, ptr, time::Instant};

use ash::vk;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{BgColor, Rect};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GpuRect {
    pos: [f32; 2],
    size: [f32; 2],
    bg_color: BgColor,
    parent_idx: u32,
    children: [u32; 2],
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

fn rects_to_gpu(rects: Vec<Rect>) -> Vec<GpuRect> {
    fn rects_to_gpu_rec(gpu_rects: &mut Vec<GpuRect>, rects: &[Rect], parent_idx: u32) {
        let start = gpu_rects.len();

        for rect in rects {
            gpu_rects.push(GpuRect {
                bg_color: rect.bg_color,
                pos: rect.pos,
                size: rect.size,
                children: [0, 0],
                parent_idx,
            });
        }

        for (gpu_rect_idx, rect) in (start..).zip(rects) {
            let start = gpu_rects.len() as u32;
            gpu_rects[gpu_rect_idx].children = [start, start + rect.children.len() as u32];
            rects_to_gpu_rec(gpu_rects, &rect.children, gpu_rect_idx as u32 + 1);
        }
    }

    let mut gpu_rects = vec![];

    rects_to_gpu_rec(&mut gpu_rects, &rects, 0);

    gpu_rects
}

pub unsafe fn unsafe_main(rectangles: &[Rect], width: u32, height: u32) {
    let rectangles = rects_to_gpu(vec![Rect {
        children: rectangles.to_vec(),
        bg_color: BgColor([1.0, 1.0, 1.0, 1.0]),
        pos: [0.0, 0.0],
        size: [width as f32, height as f32],
    }]);

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
                &vk::DeviceCreateInfo::default().queue_create_infos(&[
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family_idx)
                        .queue_priorities(&[1.0]),
                ]),
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
                        vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST,
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

    let rect_buffer = {
        let size = size_of_val(&rectangles[..]) as u64;
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
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

        let ptr = device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .unwrap();
        ptr::copy_nonoverlapping(rectangles.as_ptr() as *mut _, ptr, size as usize);
        device.unmap_memory(memory);

        buffer
    };

    let queue = device.get_device_queue(queue_family_idx, 0);

    let rect_buffer = {
        let size = size_of_val(&rectangles[..]) as u64;
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
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
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
            rect_buffer,
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

    let compute_shader = {
        device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(bytemuck::cast_slice(include_bytes!("./shader.comp.spv"))),
                None,
            )
            .unwrap()
    };

    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1),
    ];

    let set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    let set_layout = device
        .create_descriptor_set_layout(&set_layout_create_info, None)
        .unwrap();

    let set_layouts = [set_layout];

    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&[vk::PushConstantRange::default()
                    .size(mem::size_of::<u32>() as u32)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)]),
            None,
        )
        .unwrap();

    let shader_entry_name = ffi::CStr::from_bytes_with_nul_unchecked(b"main\0");
    let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .module(compute_shader)
        .name(shader_entry_name)
        .stage(vk::ShaderStageFlags::COMPUTE);

    let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage_create_info)
        .layout(pipeline_layout);

    let mut pipelines = device
        .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info], None)
        .unwrap()
        .into_iter();

    let pipeline = pipelines.next().unwrap();

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
        },
    ];

    let descriptor_pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(2)
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
        &[
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfo::default()
                    .buffer(rect_buffer)
                    .offset(0)
                    .range(size_of_val(&rectangles[..]) as u64)]),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&[vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(view)]),
        ],
        &[],
    );

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
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            )
            .image(image)],
    );

    device.cmd_clear_color_image(
        command_buffer,
        image,
        vk::ImageLayout::GENERAL,
        &vk::ClearColorValue {
            uint32: [0, 0, 0, 0],
        },
        &[vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1)],
    );

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            )
            .image(image)],
    );

    // device.cmd_bind_pipeline(
    //     command_buffer,
    //     vk::PipelineBindPoint::COMPUTE,
    //     raster_pipeline,
    // );

    // device.cmd_bind_descriptor_sets(
    //     command_buffer,
    //     vk::PipelineBindPoint::COMPUTE,
    //     pipeline_layout,
    //     0,
    //     &[descriptor_set],
    //     &[],
    // );

    // device.cmd_push_constants(
    //     command_buffer,
    //     pipeline_layout,
    //     vk::ShaderStageFlags::COMPUTE,
    //     0,
    //     bytemuck::bytes_of(&(rectangles.len() as u32)),
    // );

    // device.cmd_dispatch(command_buffer, rectangles.len() as u32, 1, 1);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            )
            .image(image)],
    );

    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[descriptor_set],
        &[],
    );

    device.cmd_push_constants(
        command_buffer,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&(rectangles.len() as u32)),
    );

    device.cmd_dispatch(command_buffer, width.div_ceil(8), height.div_ceil(8), 1);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
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

    // println!("{:?}", image_buffer);

    image::save_buffer(
        "image.png",
        &image_buffer,
        width,
        height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}
