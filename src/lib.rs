use ash::{
    khr::{surface, swapchain},
    vk,
};
use etagere::{euclid::Size2D, Size};
use rect_storage::RefIdx;
use std::{cell::RefCell, collections::HashMap, error::Error, ffi::CStr, mem, ptr, time::Instant};
use winit::{
    event::{ElementState, WindowEvent},
    event_loop::EventLoopProxy,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::WindowAttributes,
};

use crate::rect_storage::RectStorage;

mod rect_storage;

pub fn run_app<A: App>(app: A) {
    let event_loop = winit::event_loop::EventLoop::with_user_event()
        .build()
        .unwrap();

    let proxy = event_loop.create_proxy();

    event_loop
        .run_app(&mut WinitApp::new(app, EventChannel { proxy }).unwrap())
        .unwrap();
}

pub use winit::keyboard::{Key, NamedKey, NativeKey};

pub trait App {
    type Event: 'static;

    fn init(&mut self, root: &Rect, width: u32, height: u32, channel: EventChannel<Self::Event>);
    fn on_key_event(&mut self, key: Key<&str>);
    fn on_resize(&mut self, width: u32, height: u32);
    fn on_event(&mut self, event: Self::Event);
}

pub struct EventChannel<E: 'static> {
    proxy: EventLoopProxy<E>,
}

impl<E: 'static> Clone for EventChannel<E> {
    fn clone(&self) -> Self {
        Self {
            proxy: self.proxy.clone(),
        }
    }
}

impl<E: 'static> EventChannel<E> {
    pub fn send_event(&self, event: E) {
        let _ = self.proxy.send_event(event);
    }
}

struct WinitApp<A: App> {
    event_channel: EventChannel<A::Event>,
    view_ctx: Option<ViewCtx>,
    vk_rects: Option<VkRects>,
    recreate_swapchain: bool,
    render_start: Option<Instant>,
    root: Rect,
    app: A,
}

impl<A: App> WinitApp<A> {
    fn new(app: A, event_channel: EventChannel<A::Event>) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            event_channel,
            view_ctx: None,
            vk_rects: None,
            recreate_swapchain: false,
            render_start: None,
            root: Rect::new(),
            app,
        })
    }

    fn flush_rects_and_request_render(&mut self) {
        STORAGE.with_borrow(|storage| {
            TLS.with_borrow_mut(|tls| {
                let vulkan_ctx = &mut tls.as_mut().unwrap().vk_ctx;

                vulkan_ctx
                    .update_rectangles(storage.allocated_span(), self.vk_rects.as_mut().unwrap());
                vulkan_ctx.window.request_redraw();
            })
        });
    }
}

struct VulkanCtx {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    queue: vk::Queue,
    frame_rendered_fence: vk::Fence,
    image_ready_semaphore: vk::Semaphore,
    surface_instance: surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_device: swapchain::Device,
    window: winit::window::Window,
}

impl VulkanCtx {
    fn new(window: winit::window::Window) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = ash::Entry::load()?;

            let extension_names: Vec<_> =
                ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())
                    .unwrap()
                    .iter()
                    .copied()
                    .collect();

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(
                        &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3),
                    )
                    .enabled_layer_names(&["VK_LAYER_KHRONOS_validation".as_ptr() as *const i8])
                    .enabled_extension_names(&extension_names),
                None,
            )?;

            let physical_device = {
                let mut physical_devices = instance.enumerate_physical_devices()?;

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

                physical_devices[0]
            };

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
                .unwrap() as u32;

            let device = instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .enabled_extension_names(&[ash::khr::swapchain::NAME.as_ptr()])
                    .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family_idx)
                        .queue_priorities(&[1.0])]),
                None,
            )?;

            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_idx)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;

            let command_buffer = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_buffer_count(1)
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
                .unwrap()[0];

            let compute_shader = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(bytemuck::cast_slice(include_bytes!("./shader.comp.spv"))),
                None,
            )?;

            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
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
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(2)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1),
                ]),
                None,
            )?;

            let pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
                None,
            )?;

            let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .module(compute_shader)
                .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                .stage(vk::ShaderStageFlags::COMPUTE);

            let pipeline = {
                let result = device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(shader_stage_create_info)
                        .layout(pipeline_layout)],
                    None,
                );
                match result {
                    Ok(pipelines) => pipelines.into_iter().next().unwrap(),
                    Err((pipelines, result)) => {
                        result.result()?;
                        pipelines.into_iter().next().unwrap()
                    }
                }
            };
            // let memory_properties = instance.get_physical_device_memory_properties(physical_device);

            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(2)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_BUFFER,
                                descriptor_count: 1,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_IMAGE,
                                descriptor_count: 2,
                            },
                        ]),
                    None,
                )
                .unwrap();

            let descriptor_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[set_layout]),
                )
                .unwrap()[0];

            let queue = device.get_device_queue(queue_family_idx, 0);

            let frame_rendered_fence = device.create_fence(
                &vk::FenceCreateInfo::default(), //.flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;

            let image_ready_semaphore =
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;

            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?;

            let surface_instance = surface::Instance::new(&entry, &instance);

            let swapchain_device = swapchain::Device::new(&instance, &device);

            Ok(Self {
                entry,
                instance,
                device,
                physical_device,
                command_pool,
                command_buffer,
                descriptor_pool,
                descriptor_set,
                pipeline,
                pipeline_layout,
                queue,
                frame_rendered_fence,
                image_ready_semaphore,
                surface_instance,
                surface,
                swapchain_device,
                window,
            })
        }
    }

    fn update_rectangles(&mut self, new_rects: &[RectData], vk_rects: &mut VkRects) {
        unsafe {
            if vk_rects.capacity < new_rects.len() as u32 {
                self.device.destroy_buffer(vk_rects.buffer, None);
                self.device.free_memory(vk_rects.memory, None);

                *vk_rects = VkRects::new(new_rects.len().next_power_of_two() as u32, &self);

                vk_rects.write_to_descriptor_set(&self.device, self.descriptor_set);
            }

            let memory = self
                .device
                .map_memory(
                    vk_rects.memory,
                    0,
                    vk_rects.capacity as u64 * mem::size_of::<VkRect>() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut VkRect;

            let mut ptr = memory;
            for rect in new_rects {
                let [r, g, b, a] = rect.bg_color;

                let rect = VkRect {
                    pos: [rect.x as f32, rect.y as f32],
                    size: [rect.width as f32, rect.height as f32],
                    bg_color: [
                        r as f32 / 255.0,
                        g as f32 / 255.0,
                        b as f32 / 255.0,
                        a as f32 / 255.0,
                    ],
                    tex_position: rect.tex_position,
                    parent_idx: rect.parent_idx,
                    children_start: rect.children_start,
                    children_end: rect.children_end,
                    fill_kind: rect.fill_kind,
                };
                *ptr = rect;
                ptr = ptr.add(1);
            }

            self.device.unmap_memory(vk_rects.memory);
        }
    }
}

struct VkRects {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: u32,
}

unsafe fn create_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    size: u32,
    usage: vk::BufferUsageFlags,
    memory_property_flags: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer = device
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .usage(usage)
                .size(size as u64 * mem::size_of::<VkRect>() as u64),
            None,
        )
        .unwrap();

    let memory_requirements = device.get_buffer_memory_requirements(buffer);
    let memory_properties = instance.get_physical_device_memory_properties(physical_device);

    let memory_type_idx = memory_properties
        .memory_types_as_slice()
        .iter()
        .enumerate()
        .find_map(|(i, props)| {
            if props.property_flags.contains(memory_property_flags)
                && memory_requirements.memory_type_bits & (1 << i) != 0
            {
                Some(i)
            } else {
                None
            }
        })
        .unwrap() as u32;

    let memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .memory_type_index(memory_type_idx)
                .allocation_size(memory_requirements.size),
            None,
        )
        .unwrap();

    device.bind_buffer_memory(buffer, memory, 0).unwrap();

    (buffer, memory)
}

impl VkRects {
    fn new(capacity: u32, vk_ctx: &VulkanCtx) -> Self {
        unsafe {
            let (buffer, memory) = create_buffer(
                &vk_ctx.device,
                &vk_ctx.instance,
                vk_ctx.physical_device,
                capacity as u32,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            Self {
                buffer,
                memory,
                capacity,
            }
        }
    }

    fn write_to_descriptor_set(&self, device: &ash::Device, descriptor_set: vk::DescriptorSet) {
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(self.buffer)
                        .range(vk::WHOLE_SIZE)])],
                &[],
            );
        }
    }
}

struct Atlas {
    image: vk::Image,
    view: vk::ImageView,
    upload_buffer: vk::Buffer,
    upload_memory: vk::DeviceMemory,
    allocator: etagere::BucketedAtlasAllocator,
}

impl Atlas {
    fn new(vk_ctx: &VulkanCtx) -> Self {
        unsafe {
            let image = vk_ctx
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .extent(vk::Extent3D {
                            width: 1024,
                            height: 1024,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE),
                    None,
                )
                .unwrap();

            let memory_requirements = vk_ctx.device.get_image_memory_requirements(image);
            let memory_properties = vk_ctx
                .instance
                .get_physical_device_memory_properties(vk_ctx.physical_device);

            let memory_type_idx = memory_properties
                .memory_types_as_slice()
                .iter()
                .enumerate()
                .find_map(|(i, props)| {
                    if props
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                        && memory_requirements.memory_type_bits & (1 << i) != 0
                    {
                        Some(i)
                    } else {
                        None
                    }
                })
                .unwrap() as u32;

            let memory = vk_ctx
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .memory_type_index(memory_type_idx)
                        .allocation_size(memory_requirements.size),
                    None,
                )
                .unwrap();

            vk_ctx.device.bind_image_memory(image, memory, 0).unwrap();

            let view = vk_ctx
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
                .unwrap();

            let (upload_buffer, upload_memory) = create_buffer(
                &vk_ctx.device,
                &vk_ctx.instance,
                vk_ctx.physical_device,
                1024 * 1024,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            vk_ctx
                .device
                .reset_command_buffer(vk_ctx.command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            vk_ctx
                .device
                .begin_command_buffer(
                    vk_ctx.command_buffer,
                    &vk::CommandBufferBeginInfo::default(),
                )
                .unwrap();
            vk_ctx.device.cmd_pipeline_barrier(
                vk_ctx.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    )],
            );
            vk_ctx
                .device
                .end_command_buffer(vk_ctx.command_buffer)
                .unwrap();
            vk_ctx
                .device
                .queue_submit(
                    vk_ctx.queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[vk_ctx.command_buffer])
                        .wait_semaphores(&[])
                        .wait_dst_stage_mask(&[])
                        .signal_semaphores(&[])],
                    vk::Fence::null(),
                )
                .unwrap();

            vk_ctx.device.queue_wait_idle(vk_ctx.queue).unwrap();

            Self {
                image,
                view,
                upload_buffer,
                upload_memory,
                allocator: etagere::BucketedAtlasAllocator::new(Size::new(1024, 1024)),
            }
        }
    }

    fn upload_glyph(
        &mut self,
        vk_ctx: &VulkanCtx,
        glyph: &[u8],
        width: u32,
        height: u32,
    ) -> [u32; 2] {
        if glyph.len() == 0 {
            return [0, 0];
        }

        unsafe {
            let memory = vk_ctx
                .device
                .map_memory(
                    self.upload_memory,
                    0,
                    glyph.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();

            ptr::copy_nonoverlapping(glyph.as_ptr(), memory as *mut u8, glyph.len());

            vk_ctx.device.unmap_memory(self.upload_memory);

            let location = self
                .allocator
                .allocate(Size2D::new(width as i32, height as i32))
                .unwrap();

            let x = location.rectangle.min.x;
            let y = location.rectangle.min.y;

            vk_ctx.device.queue_wait_idle(vk_ctx.queue).unwrap();

            vk_ctx
                .device
                .reset_command_buffer(vk_ctx.command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            vk_ctx
                .device
                .begin_command_buffer(
                    vk_ctx.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            vk_ctx.device.cmd_copy_buffer_to_image(
                vk_ctx.command_buffer,
                self.upload_buffer,
                self.image,
                vk::ImageLayout::GENERAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x, y, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })],
            );

            vk_ctx
                .device
                .end_command_buffer(vk_ctx.command_buffer)
                .unwrap();

            vk_ctx
                .device
                .queue_submit(
                    vk_ctx.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[vk_ctx.command_buffer])],
                    vk::Fence::null(),
                )
                .unwrap();

            vk_ctx.device.queue_wait_idle(vk_ctx.queue).unwrap();

            [x as u32, y as u32]
        }
    }

    fn write_to_descriptor_set(&self, device: &ash::Device, descriptor_set: vk::DescriptorSet) {
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_view(self.view)
                        .image_layout(vk::ImageLayout::GENERAL)])],
                &[],
            );
        }
    }
}

struct ViewCtx {
    swapchain: vk::SwapchainKHR,
    frames: Vec<FrameCtx>,
}

impl ViewCtx {
    fn new(
        ctx: &VulkanCtx,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Self, Box<dyn Error>> {
        let start = Instant::now();

        unsafe {
            let surface_format = ctx
                .surface_instance
                .get_physical_device_surface_formats(ctx.physical_device, ctx.surface)
                .unwrap()
                .into_iter()
                .find(|sf| sf.format == vk::Format::R8G8B8A8_UNORM)
                .unwrap();

            eprintln!("surface format: {:.2?}", start.elapsed());

            let surface_capabilities = ctx
                .surface_instance
                .get_physical_device_surface_capabilities(ctx.physical_device, ctx.surface)?;

            eprintln!("surface capabilities: {:.2?}", start.elapsed());

            if !surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                return Err(Box::from(
                    "surface identity transform unsupported".to_string(),
                ));
            }

            let surface_resolution = {
                let current = surface_capabilities.current_extent;
                let min = surface_capabilities.min_image_extent;
                let max = surface_capabilities.max_image_extent;

                match (current.width, current.height) {
                    (u32::MAX, u32::MAX) => vk::Extent2D {
                        width: 1000.clamp(min.width, max.width),
                        height: 1000.clamp(min.height, max.height),
                    },
                    _ => current,
                }
            };

            let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(ctx.surface)
                .min_image_count(surface_capabilities.min_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .image_array_layers(1);

            if let Some(old_swapchain) = old_swapchain {
                swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain);
            }

            let swapchain = ctx
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)?;

            eprintln!("swapchain created: {:.2?}", start.elapsed());

            let images = ctx
                .swapchain_device
                .get_swapchain_images(swapchain)
                .unwrap();

            eprintln!("got images: {:.2?}", start.elapsed());

            let frames = images
                .into_iter()
                .map(|image| {
                    let view = ctx
                        .device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(image)
                                .format(surface_format.format)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .level_count(1)
                                        .layer_count(1)
                                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                                ),
                            None,
                        )
                        .unwrap();

                    let rendered = ctx
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap();

                    FrameCtx {
                        image,
                        view,
                        rendered,
                    }
                })
                .collect();

            eprintln!("got frames: {:.2?}", start.elapsed());

            Ok(Self { swapchain, frames })
        }
    }

    pub fn destroy(self, ctx: &VulkanCtx) -> vk::SwapchainKHR {
        unsafe {
            for view in self.frames.iter() {
                ctx.device.destroy_image_view(view.view, None);
                ctx.device.destroy_semaphore(view.rendered, None);
            }

            self.swapchain
        }
    }
}

struct FrameCtx {
    image: vk::Image,
    view: vk::ImageView,
    rendered: vk::Semaphore,
}

impl<A: App> winit::application::ApplicationHandler<A::Event> for WinitApp<A> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default())
            .unwrap();

        let size = window.inner_size();

        let vulkan_ctx = VulkanCtx::new(window).unwrap();
        let view_ctx = ViewCtx::new(&vulkan_ctx, None).unwrap();
        let mut vk_rects = VkRects::new(1024, &vulkan_ctx);
        vk_rects.write_to_descriptor_set(&vulkan_ctx.device, vulkan_ctx.descriptor_set);

        let atlas = Atlas::new(&vulkan_ctx);

        atlas.write_to_descriptor_set(&vulkan_ctx.device, vulkan_ctx.descriptor_set);

        TLS.replace(Some(Tls {
            atlas,
            vk_ctx: vulkan_ctx,
            text_renderer: TextRenderer::new(),
        }));

        self.app.init(
            &self.root,
            size.width,
            size.height,
            self.event_channel.clone(),
        );

        TLS.with_borrow_mut(|tls| {
            STORAGE.with_borrow(|storage| {
                tls.as_mut()
                    .unwrap()
                    .vk_ctx
                    .update_rectangles(storage.allocated_span(), &mut vk_rects)
            });
        });

        self.vk_rects = Some(vk_rects);
        self.view_ctx = Some(view_ctx);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => unsafe {
                if self.render_start.is_none() {
                    self.render_start = Some(Instant::now());
                }

                TLS.with_borrow(|tls| {
                    let vulkan_ctx = &tls.as_ref().unwrap().vk_ctx;

                    if self.recreate_swapchain {
                        if let Some(render_start) = self.render_start {
                            let elapsed = render_start.elapsed();
                            eprintln!("recreating swapchain at {:.2?}", elapsed);
                        }

                        vulkan_ctx.device.device_wait_idle().unwrap();

                        if let Some(render_start) = self.render_start {
                            let elapsed = render_start.elapsed();
                            eprintln!("wait idle done at {:.2?}", elapsed);
                        }

                        let old_swapchain = self.view_ctx.take().unwrap().destroy(vulkan_ctx);

                        if let Some(render_start) = self.render_start {
                            let elapsed = render_start.elapsed();
                            eprintln!("old view ctx destroyed at {:.2?}", elapsed);
                        }

                        self.view_ctx =
                            Some(ViewCtx::new(vulkan_ctx, Some(old_swapchain)).unwrap());

                        if let Some(render_start) = self.render_start {
                            let elapsed = render_start.elapsed();
                            eprintln!("new view ctx created at {:.2?}", elapsed);
                        }

                        self.recreate_swapchain = false;
                    }

                    let view_ctx = self.view_ctx.as_ref().unwrap();
                    let device = &vulkan_ctx.device;
                    let command_buffer = vulkan_ctx.command_buffer;

                    device
                        .reset_command_buffer(
                            vulkan_ctx.command_buffer,
                            vk::CommandBufferResetFlags::empty(),
                        )
                        .unwrap();

                    let result = vulkan_ctx.swapchain_device.acquire_next_image(
                        view_ctx.swapchain,
                        u64::MAX,
                        vulkan_ctx.image_ready_semaphore,
                        vk::Fence::null(),
                    );

                    let idx = match result {
                        Ok((idx, false)) => idx,
                        Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                            self.recreate_swapchain = true;
                            vulkan_ctx.window.request_redraw();
                            return;
                            // device.device_wait_idle().unwrap();

                            // self.view_ctx = Some(ViewCtx::new(&vulkan_ctx).unwrap());
                            // view_ctx = self.view_ctx.as_ref().unwrap();
                        }
                        Err(err) => Err(err).unwrap(),
                    };

                    let frame = &view_ctx.frames[idx as usize];

                    vulkan_ctx.device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(vulkan_ctx.descriptor_set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&[vk::DescriptorImageInfo::default()
                                .image_view(frame.view)
                                .image_layout(vk::ImageLayout::GENERAL)])],
                        &[],
                    );

                    let [width, height]: [u32; 2] = vulkan_ctx.window.inner_size().into();

                    device
                        .begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::default()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )
                        .unwrap();

                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::default(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::default()
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .image(frame.image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(1),
                            )],
                    );

                    device.cmd_clear_color_image(
                        command_buffer,
                        frame.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::default(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::GENERAL)
                            .image(frame.image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(1),
                            )],
                    );

                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        vulkan_ctx.pipeline,
                    );

                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        vulkan_ctx.pipeline_layout,
                        0,
                        &[vulkan_ctx.descriptor_set],
                        &[],
                    );

                    device.cmd_dispatch(command_buffer, width.div_ceil(32), height.div_ceil(32), 1);

                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::default(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::NONE)
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .image(frame.image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(1),
                            )],
                    );

                    device.end_command_buffer(command_buffer).unwrap();

                    device
                        .queue_submit(
                            vulkan_ctx.queue,
                            &[vk::SubmitInfo::default()
                                .command_buffers(&[command_buffer])
                                .wait_semaphores(&[vulkan_ctx.image_ready_semaphore])
                                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                                .signal_semaphores(&[frame.rendered])],
                            vulkan_ctx.frame_rendered_fence,
                        )
                        .unwrap();

                    let result = vulkan_ctx.swapchain_device.queue_present(
                        vulkan_ctx.queue,
                        &vk::PresentInfoKHR::default()
                            .image_indices(&[idx])
                            .swapchains(&[view_ctx.swapchain])
                            .wait_semaphores(&[frame.rendered]),
                    );

                    match result {
                        Ok(false) => {}
                        Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                            self.recreate_swapchain = true;
                            vulkan_ctx.window.request_redraw();

                            device
                                .wait_for_fences(&[vulkan_ctx.frame_rendered_fence], true, u64::MAX)
                                .unwrap();

                            device
                                .reset_fences(&[vulkan_ctx.frame_rendered_fence])
                                .unwrap();

                            if let Some(render_start) = self.render_start {
                                let elapsed = render_start.elapsed();
                                eprintln!("discarding render at {:.2?}", elapsed);
                            }

                            return;
                        }
                        Err(err) => Err(err).unwrap(),
                    }

                    device
                        .wait_for_fences(&[vulkan_ctx.frame_rendered_fence], true, u64::MAX)
                        .unwrap();

                    device
                        .reset_fences(&[vulkan_ctx.frame_rendered_fence])
                        .unwrap();

                    if let Some(render_start) = self.render_start {
                        let elapsed = render_start.elapsed();
                        eprintln!("render time: {:.2?}", elapsed);
                        self.render_start = None;
                    }
                });
            },
            WindowEvent::Resized(size) => {
                if self.vk_rects.is_some() {
                    self.app.on_resize(size.width, size.height);
                }
                self.flush_rects_and_request_render();
            }
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {
                if event.state != ElementState::Pressed {
                    return;
                }

                self.app.on_key_event(event.logical_key.as_ref());

                self.flush_rects_and_request_render();
            }
            _ => {}
        }
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: A::Event) {
        self.app.on_event(event);
        self.flush_rects_and_request_render();
    }
}

pub fn debug() {
    STORAGE.with_borrow(|s| eprintln!("{:?}", s.allocated_span()));
    STORAGE.with_borrow(|s| eprintln!("{:?}", s.allocated_ref_span()));
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct VkRect {
    pos: [f32; 2],
    size: [f32; 2],
    bg_color: [f32; 4],
    tex_position: [u32; 2],
    parent_idx: u32,
    children_start: u32,
    children_end: u32,
    fill_kind: u32,
}

pub struct TextLine {
    pub rect: Rect,
    clusters: Vec<Cluster>,
}

impl TextLine {
    pub fn new() -> Self {
        TextLine {
            rect: Rect::new(),
            clusters: vec![],
        }
    }

    pub fn set_text(&mut self, text: &str) {
        let line = TLS.with_borrow_mut(|tls| {
            let tls = tls.as_mut().unwrap();
            tls.text_renderer
                .get_glyphs(text, &mut tls.atlas, &mut tls.vk_ctx)
        });

        let height = 40.0;

        let right = line
            .glyphs
            .iter()
            .map(|glyph| glyph.left + glyph.width as i32)
            .max()
            .unwrap_or(0);

        let left = line
            .glyphs
            .iter()
            .map(|glyph| glyph.left)
            .min()
            .unwrap_or(0);

        self.rect.clear_children();

        for glyph in &line.glyphs {
            let child = self.rect.new_child();
            // dbg!(&child);
            child.set_size(glyph.width as u32, glyph.height as u32);
            child.set_pos(
                glyph.left as u32,
                (((height + line.x_height) / 2.0) as i32 - glyph.top) as usize as u32,
            );
            // dbg!(glyph.texture_x, glyph.texture_y);
            child.set_tex_position([glyph.texture_x, glyph.texture_y]);
        }

        self.rect
            .set_size((right - left) as u32 + 10, height as u32);

        self.clusters = line.clusters;
    }

    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }
}

struct TextRenderer {
    shape_context: swash::shape::ShapeContext,
    scale_context: swash::scale::ScaleContext,
    glyph_cache: HashMap<u16, CachedGlyph>,
}

#[derive(Clone, Copy)]
struct CachedGlyph {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
    texture_x: u32,
    texture_y: u32,
}

pub struct Cluster {
    pub start: u32,
    pub end: u32,
    pub px_offset: u32,
}

struct RenderedLine {
    glyphs: Vec<CachedGlyph>,
    clusters: Vec<Cluster>,
    x_height: f32,
}

impl TextRenderer {
    pub fn new() -> Self {
        Self {
            shape_context: swash::shape::ShapeContext::new(),
            scale_context: swash::scale::ScaleContext::new(),
            glyph_cache: HashMap::new(),
        }
    }

    pub fn get_glyphs(
        &mut self,
        text: &str,
        atlas: &mut Atlas,
        vk_ctx: &VulkanCtx,
    ) -> RenderedLine {
        let size = 30.0;
        let font = swash::FontRef::from_index(include_bytes!("../ARIAL.TTF"), 0).unwrap();

        let mut shaper = self.shape_context.builder(font).size(size).build();

        let x_height = shaper.metrics().x_height;

        let mut scaler = self
            .scale_context
            .builder(font)
            .size(size)
            .hint(true)
            .build();

        shaper.add_str(text);

        let mut glyphs = vec![];
        let mut clusters = vec![];

        let mut advance: f32 = 0.0;
        let mut glyph_idx = 0;

        shaper.shape_with(|cluster| {
            // let start = glyph_idx;
            for glyph in cluster.glyphs {
                glyph_idx += 1;
                let (mut cached_glyph, x, y) = 'glyph: {
                    // let (x, x_sub) = split(glyph.x + advance);
                    // let (y, y_sub) = split(glyph.y);

                    let x = (glyph.x + advance) as i32;
                    let y = glyph.y as i32;

                    advance += glyph.advance;

                    if let Some(cached_glyph) = self.glyph_cache.get(&glyph.id) {
                        break 'glyph (*cached_glyph, x, y);
                    }

                    // use swash::zeno::{Format, Vector};
                    // let offset = Vector::new(x_sub.as_f32(), y_sub.as_f32());

                    let Some(image) = swash::scale::Render::new(&[swash::scale::Source::Outline])
                        .format(swash::zeno::Format::Subpixel)
                        // .offset(offset)
                        .render(&mut scaler, glyph.id)
                    else {
                        panic!("No glyph");
                    };

                    let width = image.placement.width as u32;
                    let height = image.placement.height as u32;

                    let [texture_x, texture_y] =
                        atlas.upload_glyph(&vk_ctx, &image.data, width, height);

                    let mut cached_glyph = CachedGlyph {
                        left: image.placement.left,
                        top: image.placement.top,
                        width,
                        height,
                        texture_x,
                        texture_y,
                    };

                    self.glyph_cache.insert(glyph.id, cached_glyph);

                    (cached_glyph, x, y)
                };
                cached_glyph.top += y;
                cached_glyph.left += x;
                glyphs.push(cached_glyph);
            }

            if cluster.components.is_empty() {
                clusters.push(Cluster {
                    start: cluster.source.start,
                    end: cluster.source.end,
                    px_offset: advance as u32,
                });
            } else {
                // clusters.push(Cluster {
                //     start: (),
                //     end: (),
                //     px_offset: (),
                // });
                panic!()
            }
        });

        // cached.width = advance;
        RenderedLine {
            glyphs,
            x_height,
            clusters,
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct RectData {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    bg_color: [u8; 4],
    tex_position: [u32; 2],
    parent_idx: u32,
    children_start: u32,
    children_end: u32,
    children_capacity: u32,
    ref_idx: u32,
    fill_kind: u32,
}

struct Tls {
    vk_ctx: VulkanCtx,
    text_renderer: TextRenderer,
    atlas: Atlas,
}

impl Tls {
    pub fn new(window: winit::window::Window) -> Self {
        let vk_ctx = VulkanCtx::new(window).unwrap();
        let atlas = Atlas::new(&vk_ctx);

        Self {
            vk_ctx,
            text_renderer: TextRenderer::new(),
            atlas,
        }
    }
}

thread_local! {
    static TLS: RefCell<Option<Tls>> = RefCell::new(None);
    static STORAGE: RefCell<RectStorage> = RefCell::new(RectStorage::new());
}

#[derive(Debug)]
pub struct Rect {
    ref_idx: RefIdx,
}

impl Rect {
    pub fn new() -> Self {
        STORAGE.with_borrow_mut(|storage| Rect {
            ref_idx: storage.new_rect(),
        })
    }

    pub fn new_child(&self) -> Self {
        STORAGE.with_borrow_mut(|storage| Rect {
            ref_idx: storage.new_child(self.ref_idx),
        })
    }

    pub fn clear_children(&self) {
        STORAGE.with_borrow_mut(|storage| storage.clear_children(self.ref_idx))
    }

    pub fn remove_from_parent(&self) {
        STORAGE.with_borrow_mut(|storage| storage.remove_from_parent(self.ref_idx))
    }

    pub fn new_text_child(&self) -> TextLine {
        let rect = self.new_child();
        TextLine {
            rect,
            clusters: vec![],
        }
    }

    pub fn set_bg_color(&self, color: [u8; 4]) {
        STORAGE.with_borrow_mut(|storage| storage.set_bg_color(self.ref_idx, color))
    }

    pub fn set_tex_position(&self, tex_position: [u32; 2]) {
        STORAGE.with_borrow_mut(|storage| storage.set_tex_position(self.ref_idx, tex_position))
    }

    pub fn set_size(&self, width: u32, height: u32) {
        STORAGE.with_borrow_mut(|storage| storage.set_size(self.ref_idx, width, height))
    }

    pub fn set_pos(&self, x: u32, y: u32) {
        STORAGE.with_borrow_mut(|storage| storage.set_pos(self.ref_idx, x, y))
    }

    pub fn get_size(&self) -> (u32, u32) {
        STORAGE.with_borrow_mut(|storage| storage.get_size(self.ref_idx))
    }

    pub fn append_child(&self, child: &Rect) {
        STORAGE.with_borrow_mut(|storage| storage.append_child(self.ref_idx, child.ref_idx))
    }

    // pub fn debug(&self) {
    //     STORAGE.with_borrow_mut(|storage| storage.debug(self))
    // }
}

impl Drop for Rect {
    fn drop(&mut self) {
        STORAGE.with_borrow_mut(|storage| {
            storage.decrement_ref_count(self.ref_idx);
        })
    }
}

impl Clone for Rect {
    fn clone(&self) -> Self {
        STORAGE.with_borrow_mut(|storage| storage.increment_ref_count(self.ref_idx));
        Rect {
            ref_idx: self.ref_idx,
        }
    }
}
