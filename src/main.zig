const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;

// HSA Runtime C bindings
const c = @cImport({
    @cInclude("hsa/hsa.h");
    @cInclude("hsa/hsa_ext_amd.h");
});

const HsaError = error{
    HsaInitFailed,
    AgentNotFound,
    QueueCreationFailed,
    MemoryAllocationFailed,
    CodeObjectLoadFailed,
    KernelNotFound,
    ExecutionFailed,
    RequiredMemoryRegionNotFound,
};

const GpuContext = struct {
    agent: c.hsa_agent_t,
    queue: ?*c.hsa_queue_t,
    kernarg_region: ?c.hsa_region_t,
    fine_grained_region: ?c.hsa_region_t,
    coarse_grained_region: ?c.hsa_region_t,

    const Self = @This();

    fn init() !Self {
        // Initialize HSA
        const status = c.hsa_init();
        if (status != c.HSA_STATUS_SUCCESS) {
            print("HSA init failed with status: {}\n", .{status});
            return HsaError.HsaInitFailed;
        }

        print("HSA initialized successfully\n", .{});

        var ctx = Self{
            .agent = undefined,
            .queue = null,
            .kernarg_region = null,
            .fine_grained_region = null,
            .coarse_grained_region = null,
        };

        print("Searching for GPU agents...\n", .{});

        // Find GPU agent
        const find_status = c.hsa_iterate_agents(findGpuAgent, &ctx.agent);
        print("Agent iteration returned: {}\n", .{find_status});

        if (find_status != c.HSA_STATUS_SUCCESS and find_status != c.HSA_STATUS_INFO_BREAK) {
            print("No GPU agent found. Status: {}\n", .{find_status});
            return HsaError.AgentNotFound;
        }

        // Find memory regions
        print("Searching for memory regions...\n", .{});
        const region_status = c.hsa_agent_iterate_regions(ctx.agent, findMemoryRegions, &ctx);
        if (region_status != c.HSA_STATUS_SUCCESS) {
            return HsaError.AgentNotFound;
        }

        // Validate that required regions were found
        print("Memory region validation:\n", .{});
        print("  Kernarg region: {}\n", .{ctx.kernarg_region != null});
        print("  Fine-grained region: {}\n", .{ctx.fine_grained_region != null});
        print("  Coarse-grained region: {}\n", .{ctx.coarse_grained_region != null});

        if (ctx.kernarg_region == null) {
            print("Note: No dedicated kernarg region found, will use fine-grained global memory\n", .{});
        }

        if (ctx.coarse_grained_region == null) {
            print("ERROR: Coarse-grained region not found!\n", .{});
            return HsaError.RequiredMemoryRegionNotFound;
        }

        if (ctx.fine_grained_region == null) {
            print("ERROR: Fine-grained region not found!\n", .{});
            return HsaError.RequiredMemoryRegionNotFound;
        }

        // Create queue
        const queue_status = c.hsa_queue_create(ctx.agent, 1024, c.HSA_QUEUE_TYPE_MULTI, null, null, 0, 0, &ctx.queue);
        if (queue_status != c.HSA_STATUS_SUCCESS) {
            return HsaError.QueueCreationFailed;
        }

        return ctx;
    }

    fn deinit(self: *Self) void {
        if (self.queue) |queue| {
            _ = c.hsa_queue_destroy(queue);
        }
        _ = c.hsa_shut_down();
    }

    fn allocateMemory(self: *Self, size: usize, comptime T: type) ![]T {
        if (self.coarse_grained_region == null) {
            print("ERROR: No coarse-grained region available for memory allocation\n", .{});
            return HsaError.MemoryAllocationFailed;
        }

        print("Allocating {} bytes ({} elements of {s}) using coarse-grained memory\n", .{ size, size / @sizeOf(T), @typeName(T) });

        var ptr: ?*anyopaque = null;
        const status = c.hsa_memory_allocate(self.coarse_grained_region.?, size, &ptr);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Memory allocation failed with status: {} (0x{X})\n", .{ status, status });
            return HsaError.MemoryAllocationFailed;
        }

        const typed_ptr: [*]T = @ptrCast(@alignCast(ptr));
        return typed_ptr[0 .. size / @sizeOf(T)];
    }

    fn allocateKernargs(self: *Self, size: usize) ![]u8 {
        const region = if (self.kernarg_region) |kr| kr else self.fine_grained_region.?;

        print("Allocating {} bytes for kernel arguments using {s} region\n", .{ size, if (self.kernarg_region != null) "kernarg" else "fine-grained" });

        var ptr: ?*anyopaque = null;
        const status = c.hsa_memory_allocate(region, size, &ptr);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Kernel argument allocation failed with status: {} (0x{X})\n", .{ status, status });
            return HsaError.MemoryAllocationFailed;
        }

        const byte_ptr: [*]u8 = @ptrCast(ptr);
        return byte_ptr[0..size];
    }

    fn freeMemory(self: *Self, ptr: anytype) void {
        _ = self;
        _ = c.hsa_memory_free(@ptrCast(ptr.ptr));
    }
};

const KernelManager = struct {
    code_object_reader: c.hsa_code_object_reader_t,
    executable: c.hsa_executable_t,
    code_data: []u8,
    allocator: std.mem.Allocator,
    agent: c.hsa_agent_t, // Add agent reference

    const Self = @This();

    fn init(ctx: *GpuContext, object_file: []const u8) !Self {
        const allocator = std.heap.page_allocator;

        print("Attempting to load kernel object file: {s}\n", .{object_file});

        // Create code object from file
        var file = std.fs.cwd().openFile(object_file, .{}) catch |err| {
            print("Failed to open kernel object file '{s}': {}\n", .{ object_file, err });
            return HsaError.CodeObjectLoadFailed;
        };
        defer file.close();

        const file_size = try file.getEndPos();
        print("Kernel object file size: {} bytes\n", .{file_size});

        const code_data = try allocator.alloc(u8, file_size);
        errdefer allocator.free(code_data);

        _ = try file.readAll(code_data);
        print("Successfully read {} bytes from kernel file\n", .{code_data.len});

        if (code_data.len >= 4) {
            print("File header: 0x{X:0>2} 0x{X:0>2} 0x{X:0>2} 0x{X:0>2}\n", .{ code_data[0], code_data[1], code_data[2], code_data[3] });
        }

        var manager = Self{
            .code_object_reader = undefined,
            .executable = undefined,
            .code_data = code_data,
            .allocator = allocator,
            .agent = ctx.agent, // Store agent reference
        };

        // Create code object reader from memory
        print("Creating code object reader from memory...\n", .{});
        var status = c.hsa_code_object_reader_create_from_memory(code_data.ptr, file_size, &manager.code_object_reader);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Failed to create code object reader: {} (0x{X})\n", .{ status, status });
            return HsaError.CodeObjectLoadFailed;
        }
        print("Code object reader created successfully\n", .{});

        // Create executable
        print("Creating HSA executable...\n", .{});
        status = c.hsa_executable_create_alt(c.HSA_PROFILE_FULL, c.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, null, &manager.executable);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Failed to create executable: {} (0x{X})\n", .{ status, status });
            _ = c.hsa_code_object_reader_destroy(manager.code_object_reader);
            return HsaError.CodeObjectLoadFailed;
        }
        print("HSA executable created successfully\n", .{});

        // Load code object reader
        print("Loading agent code object...\n", .{});
        status = c.hsa_executable_load_agent_code_object(manager.executable, ctx.agent, manager.code_object_reader, null, null);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Failed to load agent code object: {} (0x{X})\n", .{ status, status });
            _ = c.hsa_executable_destroy(manager.executable);
            _ = c.hsa_code_object_reader_destroy(manager.code_object_reader);
            return HsaError.CodeObjectLoadFailed;
        }
        print("Agent code object loaded successfully\n", .{});

        // Freeze executable
        print("Freezing executable...\n", .{});
        status = c.hsa_executable_freeze(manager.executable, null);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Failed to freeze executable: {} (0x{X})\n", .{ status, status });
            _ = c.hsa_executable_destroy(manager.executable);
            _ = c.hsa_code_object_reader_destroy(manager.code_object_reader);
            return HsaError.CodeObjectLoadFailed;
        }
        print("Executable frozen successfully\n", .{});

        return manager;
    }

    fn getKernelSymbol(self: *Self, kernel_name: []const u8) !c.hsa_executable_symbol_t {
        var symbol: c.hsa_executable_symbol_t = undefined;

        // Try the .kd suffix approach first
        var kernel_name_buffer: [256]u8 = undefined;
        const full_name = std.fmt.bufPrintZ(&kernel_name_buffer, "{s}.kd", .{kernel_name}) catch {
            print("Kernel name too long: {s}\n", .{kernel_name});
            return HsaError.KernelNotFound;
        };

        print("Looking for kernel symbol: {s} (length: {})\n", .{ full_name, full_name.len });

        // FIX: Pass the agent reference instead of null
        var status = c.hsa_executable_get_symbol_by_name(self.executable, full_name.ptr, &self.agent, &symbol);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Symbol lookup with .kd failed: {} (0x{X})\n", .{ status, status });

            // Try without .kd suffix
            const base_name = std.fmt.bufPrintZ(&kernel_name_buffer, "{s}", .{kernel_name}) catch {
                return HsaError.KernelNotFound;
            };
            print("Trying base name: {s}\n", .{base_name});

            status = c.hsa_executable_get_symbol_by_name(self.executable, base_name.ptr, &self.agent, &symbol);
            if (status != c.HSA_STATUS_SUCCESS) {
                print("Base name lookup also failed: {} (0x{X})\n", .{ status, status });

                // Show available symbols for debugging
                print("Available symbols:\n", .{});
                _ = c.hsa_executable_iterate_symbols(self.executable, listSymbolsCallback, null);

                return HsaError.KernelNotFound;
            }
        }

        print("Successfully found kernel symbol!\n", .{});
        return symbol;
    }

    fn deinit(self: *Self) void {
        _ = c.hsa_executable_destroy(self.executable);
        _ = c.hsa_code_object_reader_destroy(self.code_object_reader);
        self.allocator.free(self.code_data);
    }
};

// Helper callback function to list symbols
fn listSymbolsCallback(executable: c.hsa_executable_t, symbol: c.hsa_executable_symbol_t, data: ?*anyopaque) callconv(.C) c.hsa_status_t {
    _ = executable;
    _ = data;

    var symbol_type: c.hsa_symbol_kind_t = undefined;
    var status = c.hsa_executable_symbol_get_info(symbol, c.HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbol_type);
    if (status != c.HSA_STATUS_SUCCESS) {
        return c.HSA_STATUS_SUCCESS;
    }

    if (symbol_type == c.HSA_SYMBOL_KIND_KERNEL) {
        var name_length: u32 = 0;
        status = c.hsa_executable_symbol_get_info(symbol, c.HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
        if (status != c.HSA_STATUS_SUCCESS) {
            return c.HSA_STATUS_SUCCESS;
        }

        var name_buffer: [256]u8 = undefined;
        if (name_length < name_buffer.len) {
            status = c.hsa_executable_symbol_get_info(symbol, c.HSA_EXECUTABLE_SYMBOL_INFO_NAME, &name_buffer);
            if (status == c.HSA_STATUS_SUCCESS) {
                const name_slice = name_buffer[0..name_length];
                print("  - Kernel: {s}\n", .{name_slice});
            }
        }
    }

    return c.HSA_STATUS_SUCCESS;
}

fn findMemoryRegions(region: c.hsa_region_t, data: ?*anyopaque) callconv(.C) c.hsa_status_t {
    const ctx: *GpuContext = @ptrCast(@alignCast(data));

    var segment: c.hsa_region_segment_t = undefined;
    const status = c.hsa_region_get_info(region, c.HSA_REGION_INFO_SEGMENT, &segment);
    if (status != c.HSA_STATUS_SUCCESS) {
        return status;
    }

    print("Found memory region with segment type: {}\n", .{segment});

    if (segment == c.HSA_REGION_SEGMENT_KERNARG) {
        print("  -> Kernarg region found!\n", .{});
        ctx.kernarg_region = region;
    } else if (segment == c.HSA_REGION_SEGMENT_GLOBAL) {
        var flags: c.hsa_region_global_flag_t = undefined;
        const flag_status = c.hsa_region_get_info(region, c.HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
        if (flag_status != c.HSA_STATUS_SUCCESS) {
            return flag_status;
        }

        print("  -> Global region with flags: {}\n", .{flags});

        if ((flags & c.HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
            print("     -> Fine-grained region found!\n", .{});
            ctx.fine_grained_region = region;
        } else {
            print("     -> Coarse-grained region found!\n", .{});
            ctx.coarse_grained_region = region;
        }
    }

    return c.HSA_STATUS_SUCCESS;
}

fn findGpuAgent(agent: c.hsa_agent_t, data: ?*anyopaque) callconv(.C) c.hsa_status_t {
    var device_type: c.hsa_device_type_t = undefined;
    const status = c.hsa_agent_get_info(agent, c.HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != c.HSA_STATUS_SUCCESS) {
        print("Failed to get agent info: {}\n", .{status});
        return status;
    }

    var name: [64]u8 = undefined;
    const name_status = c.hsa_agent_get_info(agent, c.HSA_AGENT_INFO_NAME, &name);
    if (name_status == c.HSA_STATUS_SUCCESS) {
        print("Found agent: {s}, type: {}\n", .{ std.mem.sliceTo(&name, 0), device_type });
    }

    if (device_type == c.HSA_DEVICE_TYPE_GPU) {
        print("Found GPU agent!\n", .{});
        const agent_ptr: *c.hsa_agent_t = @ptrCast(@alignCast(data));
        agent_ptr.* = agent;
        return c.HSA_STATUS_INFO_BREAK;
    }

    return c.HSA_STATUS_SUCCESS;
}

fn executeKernel(
    ctx: *GpuContext,
    symbol: c.hsa_executable_symbol_t,
    kernargs: []const u8,
    grid_size: [3]u32,
    workgroup_size: [3]u32,
) !void {
    var kernel_object: u64 = undefined;
    const status = c.hsa_executable_symbol_get_info(symbol, c.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
    if (status != c.HSA_STATUS_SUCCESS) {
        return HsaError.ExecutionFailed;
    }

    const queue = ctx.queue.?;
    const packet_id = c.hsa_queue_add_write_index_relaxed(queue, 1);

    const base_packets = @as([*]c.hsa_kernel_dispatch_packet_t, @ptrCast(@alignCast(queue.base_address)));
    const packet_ptr = &base_packets[@mod(packet_id, queue.size)];

    packet_ptr.header = @as(u16, c.HSA_PACKET_TYPE_KERNEL_DISPATCH) << c.HSA_PACKET_HEADER_TYPE |
        @as(u16, c.HSA_FENCE_SCOPE_SYSTEM) << c.HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
        @as(u16, c.HSA_FENCE_SCOPE_SYSTEM) << c.HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    packet_ptr.setup = 1;
    packet_ptr.workgroup_size_x = @intCast(workgroup_size[0]);
    packet_ptr.workgroup_size_y = @intCast(workgroup_size[1]);
    packet_ptr.workgroup_size_z = @intCast(workgroup_size[2]);
    packet_ptr.grid_size_x = @intCast(grid_size[0]);
    packet_ptr.grid_size_y = @intCast(grid_size[1]);
    packet_ptr.grid_size_z = @intCast(grid_size[2]);
    packet_ptr.kernel_object = kernel_object;
    packet_ptr.kernarg_address = @ptrCast(@constCast(kernargs.ptr));
    packet_ptr.private_segment_size = 0;
    packet_ptr.group_segment_size = 2048;

    c.hsa_queue_store_write_index_relaxed(queue, packet_id + 1);
    c.hsa_signal_store_relaxed(queue.doorbell_signal, @intCast(packet_id));

    while (c.hsa_signal_wait_scacquire(queue.doorbell_signal, c.HSA_SIGNAL_CONDITION_LT, @intCast(packet_id + 1), std.math.maxInt(u64), c.HSA_WAIT_STATE_BLOCKED) != 0) {}
}

pub fn main() !void {
    print("Initializing AMD GPU compute with HSA runtime...\n", .{});

    var ctx = GpuContext.init() catch |err| {
        print("Failed to initialize GPU context: {}\n", .{err});
        return;
    };
    defer ctx.deinit();

    print("GPU context initialized successfully\n", .{});

    var kernel_manager = KernelManager.init(&ctx, "kernel.o") catch |err| {
        print("Failed to load kernels: {}\n", .{err});
        return;
    };
    defer kernel_manager.deinit();

    print("Kernels loaded successfully\n", .{});

    try testVectorAddition(&ctx, &kernel_manager);
    try testMatrixMultiplication(&ctx, &kernel_manager);

    print("All tests completed successfully!\n", .{});
}

fn testVectorAddition(ctx: *GpuContext, kernel_manager: *KernelManager) !void {
    print("\n=== Testing Vector Addition with Shared Memory ===\n", .{});

    const n: u32 = 1024;
    const workgroup_size: u32 = 256;
    const num_groups = (n + workgroup_size - 1) / workgroup_size;

    const a = try ctx.allocateMemory(n * @sizeOf(f32), f32);
    const b = try ctx.allocateMemory(n * @sizeOf(f32), f32);
    const result_c = try ctx.allocateMemory(num_groups * @sizeOf(f32), f32);
    defer ctx.freeMemory(a);
    defer ctx.freeMemory(b);
    defer ctx.freeMemory(result_c);

    for (0..n) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i * 2);
    }

    const kernargs = try ctx.allocateKernargs(4 * 8);
    defer ctx.freeMemory(kernargs);

    const arg_ptrs = std.mem.bytesAsSlice(u64, kernargs);
    arg_ptrs[0] = @intFromPtr(a.ptr);
    arg_ptrs[1] = @intFromPtr(b.ptr);
    arg_ptrs[2] = @intFromPtr(result_c.ptr);
    arg_ptrs[3] = n;

    const symbol = try kernel_manager.getKernelSymbol("vector_add_shared");
    try executeKernel(ctx, symbol, kernargs, .{ num_groups * workgroup_size, 1, 1 }, .{ workgroup_size, 1, 1 });

    print("First few results: ", .{});
    for (0..@min(8, num_groups)) |i| {
        print("{:.1} ", .{result_c[i]});
    }
    print("\n", .{});
}

fn testMatrixMultiplication(ctx: *GpuContext, kernel_manager: *KernelManager) !void {
    print("\n=== Testing Matrix Multiplication with Shared Memory ===\n", .{});

    const width: u32 = 128;
    const tile_size: u32 = 16;

    const a = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    const b = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    const result_c = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    defer ctx.freeMemory(a);
    defer ctx.freeMemory(b);
    defer ctx.freeMemory(result_c);

    for (0..width) |i| {
        for (0..width) |j| {
            a[i * width + j] = @floatFromInt(i + j);
            b[i * width + j] = @floatFromInt(i * j + 1);
            result_c[i * width + j] = 0.0;
        }
    }

    const kernargs = try ctx.allocateKernargs(4 * 8);
    defer ctx.freeMemory(kernargs);

    const arg_ptrs = std.mem.bytesAsSlice(u64, kernargs);
    arg_ptrs[0] = @intFromPtr(a.ptr);
    arg_ptrs[1] = @intFromPtr(b.ptr);
    arg_ptrs[2] = @intFromPtr(result_c.ptr);
    arg_ptrs[3] = width;

    const symbol = try kernel_manager.getKernelSymbol("matrix_multiply_shared");
    const grid_dim = (width + tile_size - 1) / tile_size;

    try executeKernel(ctx, symbol, kernargs, .{ grid_dim * tile_size, grid_dim * tile_size, 1 }, .{ tile_size, tile_size, 1 });

    print("Sample results from C matrix:\n", .{});
    for (0..4) |i| {
        for (0..4) |j| {
            print("{:.1} ", .{result_c[i * width + j]});
        }
        print("\n", .{});
    }
}
