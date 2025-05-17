use std::{cell::RefCell, error::Error, ffi::CStr, mem, ptr, time::Instant};

use ash::{
    khr::{surface, swapchain},
    vk,
};
use rect_allocator::Allocator;
use ref_allocator::{Ref, RefAllocator};
use winit::{
    event::WindowEvent,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::WindowAttributes,
};

mod rect_allocator;
mod ref_allocator;

fn main() {
    let start = Instant::now();
    let rect = Rect::new();
    rect.new_child();
    rect.new_child();
    rect.new_child();
    let child = rect.new_child();
    rect.new_child();

    child.new_child();
    child.new_child();
    child.new_child();
    child.new_child();
    child.new_child();
    child.new_child();
    child.new_child();
    child.new_child();
    child.new_child();
    child.new_child();
    child.new_child();

    eprintln!("{:?}", start.elapsed());
    rect.debug();

    winit::event_loop::EventLoop::builder()
        .build()
        .unwrap()
        .run_app(app)
        .unwrap();
}

struct App {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    rect_buffer: vk::Buffer,
    surface: Option<vk::SurfaceKHR>,
}

impl App {
    fn new() -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = ash::Entry::load()?;

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(
                        &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3),
                    )
                    .enabled_layer_names(&["VK_LAYER_KHRONOS_validation".as_ptr() as *const i8]),
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

                physical_devices
                    .get(0)
                    .copied()
                    .ok_or_else(|| Box::from("No physical device found".to_string()))?
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
                .ok_or_else(|| Box::from("No queue family found".to_string()))?
                as u32;

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
                &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_idx),
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
                ]),
                None,
            )?;

            let pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[set_layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .size(mem::size_of::<u32>() as u32)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)]),
                None,
            )?;

            let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .module(compute_shader)
                .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                .stage(vk::ShaderStageFlags::COMPUTE);

            let mut pipeline = {
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
                                descriptor_count: 1,
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

            Ok(Self {
                entry,
                instance,
                device,
                physical_device,
                command_pool,
                command_buffer,
                descriptor_pool,
                descriptor_set,
                rect_buffer: None,
                surface: None,
            })
        }
    }
}

struct ConstViewStuff {
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    swapchain_device: swapchain::Device,
}

impl ConstViewStuff {
    fn new(app: &App, window: &winit::window::Window) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let surface = ash_window::create_surface(
                &app.entry,
                &app.instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?;

            let surface_instance = surface::Instance::new(&app.entry, &app.instance);

            let surface_format = surface_instance
                .get_physical_device_surface_formats(app.physical_device, surface)?;
            let surface_capabilities = surface_instance
                .get_physical_device_surface_capabilities(app.physical_device, surface)?;

            if surface_capabilities.min_image_count > 1 {
                return Err(Box::from(
                    "single swapchain image is unsupported".to_string(),
                ));
            }

            let surface_resolution = match surface_capabilities.current_extent.width {
                u32::MAX => vk::Extent2D {
                    width: 1024,
                    height: 1024,
                },
                _ => surface_capabilities.current_extent,
            };

            if !surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                return Err(Box::from(
                    "surface identity transform unsupported".to_string(),
                ));
            }

            let swapchain_device = swapchain::Device::new(&app.instance, &app.device);

            Self {}
        }
    }
}

struct ViewStuff {
    swapchain: vk::SwapchainKHR,
    image: vk::Image,
    view: vk::ImageView,
}

impl ViewStuff {
    fn new(app: &App, const_stuff: &ConstViewStuff) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let surface_resolution = {
                let [width, height]: [u32; 2] = window.inner_size().into();
                vk::Extent2D { width, height }
            };

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(const_stuff.surface)
                .min_image_count(1)
                .image_color_space(const_stuff.surface_format.color_space)
                .image_format(const_stuff.surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::STORAGE)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::IMMEDIATE)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = const_stuff
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)?;

            let image = const_stuff
                .swapchain_device
                .get_swapchain_images(swapchain)?
                .into_iter()
                .next()
                .unwrap();

            Self {
                swapchain: (),
                image: (),
                view: (),
            }
        }
    }
}

impl winit::application::ApplicationHandler<()> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default())
            .unwrap();

        let surface = unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
        }
        .unwrap();

        self.surface = Some(surface);
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
                let device = self.device;
                let command_buffer = self.command_buffer;

                device
                    .begin_command_buffer(
                        command_buffer,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();

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
            },
            _ => {}
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RectData {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub bg_color: [u8; 4],
    pub parent_idx: u32,
    pub children_start: u32,
    pub children_end: u32,
    pub children_capacity: u32,
    pub ref_idx: u32,
}

struct RectStorage {
    ref_allocator: RefAllocator,
    rect_allocator: Allocator<RectData>,
}

impl RectStorage {
    fn new() -> Self {
        Self {
            ref_allocator: RefAllocator::new(),
            rect_allocator: Allocator::new(),
        }
    }

    pub fn new_rect(&mut self) -> Rect {
        let ref_idx = self.ref_allocator.alloc();
        let rect_idx = self.rect_allocator.alloc(1);

        *self.ref_allocator.resolve(ref_idx) = Ref {
            rect_idx,
            ref_count: 1,
        };
        unsafe {
            self.rect_allocator.resolve(rect_idx).write(RectData {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                bg_color: [0, 0, 0, 0],
                parent_idx: 0,
                children_start: 0,
                children_end: 0,
                children_capacity: 0,
                ref_idx: ref_idx + 1,
            });
        }

        Rect { ref_idx }
    }

    pub fn new_child(&mut self, rect: &Rect) -> Rect {
        // eprintln!("new child");
        let ref_idx = self.ref_allocator.alloc();

        let (start, end, cap) = unsafe {
            let rect = self.resolve_rect(rect.ref_idx).read();

            let child_count = rect.children_end - rect.children_start;

            if child_count == rect.children_capacity {
                // eprintln!("new child - realloc");
                let new_length = if child_count == 0 { 4 } else { child_count * 2 };
                let new_children_start = self.rect_allocator.alloc(new_length);
                let mut new_children = self.rect_allocator.resolve(new_children_start);
                let old_children = self.rect_allocator.resolve(rect.children_start);
                // eprintln!("copying {child_count} children");
                ptr::copy_nonoverlapping(old_children, new_children, child_count as usize);
                if child_count != 0 {
                    self.rect_allocator.free(rect.children_start, child_count);
                }
                for idx in new_children_start..(new_children_start + child_count) {
                    let child = &mut *new_children;
                    if let Some(ref_idx) = child.ref_idx.checked_sub(1) {
                        // eprintln!("updating ref {ref_idx}");
                        self.ref_allocator.resolve(ref_idx).rect_idx = idx;
                    }

                    for child_child_idx in child.children_start..child.children_end {
                        let child_child = &mut *self.rect_allocator.resolve(child_child_idx);
                        child_child.parent_idx = idx + 1;
                    }

                    new_children = new_children.add(1);
                }

                (
                    new_children_start,
                    new_children_start + child_count,
                    new_length,
                )
            } else {
                (
                    rect.children_start,
                    rect.children_end,
                    rect.children_capacity,
                )
            }
        };

        unsafe {
            let rect_idx = self.ref_allocator.resolve(rect.ref_idx).rect_idx;
            let rect = &mut *self.rect_allocator.resolve(rect_idx);

            rect.children_start = start;
            rect.children_end = end + 1;
            rect.children_capacity = cap;

            self.rect_allocator.resolve(end).write(RectData {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                bg_color: [0, 0, 0, 0],
                parent_idx: rect_idx + 1,
                children_start: 0,
                children_end: 0,
                children_capacity: 0,
                ref_idx: ref_idx + 1,
            });
        }

        *self.ref_allocator.resolve(ref_idx) = Ref {
            rect_idx: end,
            ref_count: 1,
        };

        Rect { ref_idx }
    }

    pub fn set_bg_color(&mut self, rect: &Rect, color: [u8; 4]) {
        unsafe {
            let rect = &mut *self.resolve_rect(rect.ref_idx);
            rect.bg_color = color;
        }
    }

    pub fn set_size(&mut self, rect: &Rect, width: u32, height: u32) {
        unsafe {
            let rect = &mut *self.resolve_rect(rect.ref_idx);
            rect.width = width;
            rect.height = height;
        }
    }

    pub fn set_pos(&mut self, rect: &Rect, x: u32, y: u32) {
        unsafe {
            let rect = &mut *self.resolve_rect(rect.ref_idx);
            rect.x = x;
            rect.y = y;
        }
    }

    pub fn debug(&mut self, rect: &Rect) {
        let rect_idx = self.ref_allocator.resolve(rect.ref_idx).rect_idx;
        self.debug_rec(rect_idx, 0);
    }

    fn debug_rec(&mut self, rect_idx: u32, depth: u32) {
        unsafe {
            let rect = self.rect_allocator.resolve(rect_idx).read();
            for _ in 0..depth {
                eprint!("  ");
            }
            // eprintln!("rect {}..{}", rect.children_start, rect.children_end);
            eprintln!("rect {rect_idx} in {:?}", rect.parent_idx.checked_sub(1));
            for child_idx in rect.children_start..rect.children_end {
                self.debug_rec(child_idx, depth + 1);
            }
        }
    }

    fn resolve_rect(&mut self, ref_idx: u32) -> *mut RectData {
        unsafe {
            self.rect_allocator
                .resolve(self.ref_allocator.resolve(ref_idx).rect_idx)
        }
    }
}

thread_local! {
    static STORAGE: RefCell<RectStorage> = RefCell::new(RectStorage::new());
}

pub struct Rect {
    ref_idx: u32,
}

impl Rect {
    pub fn new() -> Self {
        STORAGE.with_borrow_mut(|storage| storage.new_rect())
    }

    pub fn new_child(&self) -> Self {
        STORAGE.with_borrow_mut(|storage| storage.new_child(self))
    }

    pub fn set_bg_color(&mut self, color: [u8; 4]) {
        STORAGE.with_borrow_mut(|storage| storage.set_bg_color(self, color))
    }

    pub fn set_size(&mut self, width: u32, height: u32) {
        STORAGE.with_borrow_mut(|storage| storage.set_size(self, width, height))
    }

    pub fn set_pos(&mut self, x: u32, y: u32) {
        STORAGE.with_borrow_mut(|storage| storage.set_pos(self, x, y))
    }

    pub fn debug(&self) {
        STORAGE.with_borrow_mut(|storage| storage.debug(self))
    }
}

impl Drop for Rect {
    fn drop(&mut self) {
        STORAGE.with_borrow_mut(|storage| {
            let ref_ = storage.ref_allocator.resolve(self.ref_idx);
            ref_.ref_count -= 1;
            if ref_.ref_count == 0 {
                unsafe {
                    let rect = &mut *storage.rect_allocator.resolve(ref_.rect_idx);
                    rect.ref_idx = 0;
                    if rect.parent_idx == 0 {
                        storage.rect_allocator.free(ref_.rect_idx, 1);
                    }
                }
            }
        })
    }
}
