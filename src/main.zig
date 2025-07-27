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
};

const GpuContext = struct {
    agent: c.hsa_agent_t,
    queue: ?*c.hsa_queue_t,
    kernarg_region: c.hsa_region_t,
    fine_grained_region: c.hsa_region_t,
    coarse_grained_region: c.hsa_region_t,

    const Self = @This();

    fn init(allocator: Allocator) !Self {
        // Initialize HSA
        var status = c.hsa_init();
        if (status != c.HSA_STATUS_SUCCESS) {
            return HsaError.HsaInitFailed;
        }

        var ctx = Self{
            .agent = undefined,
            .queue = null,
            .kernarg_region = undefined,
            .fine_grained_region = undefined,
            .coarse_grained_region = undefined,
        };

        // Find GPU agent
        status = c.hsa_iterate_agents(findGpuAgent, &ctx.agent);
        if (status != c.HSA_STATUS_SUCCESS) {
            return HsaError.AgentNotFound;
        }

        // Find memory regions
        status = c.hsa_agent_iterate_regions(ctx.agent, findMemoryRegions, &ctx);
        if (status != c.HSA_STATUS_SUCCESS) {
            return HsaError.AgentNotFound;
        }

        // Create queue
        status = c.hsa_queue_create(ctx.agent, 1024, c.HSA_QUEUE_TYPE_MULTI, null, null, 0, 0, &ctx.queue);
        if (status != c.HSA_STATUS_SUCCESS) {
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
        var ptr: ?*anyopaque = null;
        const status = c.hsa_memory_allocate(self.coarse_grained_region, size, &ptr);
        if (status != c.HSA_STATUS_SUCCESS) {
            return HsaError.MemoryAllocationFailed;
        }

        const typed_ptr: [*]T = @ptrCast(@alignCast(ptr));
        return typed_ptr[0 .. size / @sizeOf(T)];
    }

    fn allocateKernargs(self: *Self, size: usize) ![]u8 {
        var ptr: ?*anyopaque = null;
        const status = c.hsa_memory_allocate(self.kernarg_region, size, &ptr);
        if (status != c.HSA_STATUS_SUCCESS) {
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

// Kernel management
const KernelManager = struct {
    code_object: c.hsa_code_object_t,
    executable: c.hsa_executable_t,

    const Self = @This();

    fn init(ctx: *GpuContext, object_file: []const u8) !Self {
        var manager = Self{
            .code_object = undefined,
            .executable = undefined,
        };

        // Create code object from file
        var file = std.fs.cwd().openFile(object_file, .{}) catch |err| {
            print("Failed to open kernel object file: {}\n", .{err});
            return HsaError.CodeObjectLoadFailed;
        };
        defer file.close();

        const file_size = try file.getEndPos();
        const allocator = std.heap.page_allocator;
        const code_data = try allocator.alloc(u8, file_size);
        defer allocator.free(code_data);

        _ = try file.readAll(code_data);

        var status = c.hsa_code_object_deserialize(code_data.ptr, file_size, null, &manager.code_object);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Failed to deserialize code object: {}\n", .{status});
            return HsaError.CodeObjectLoadFailed;
        }

        // Create executable
        status = c.hsa_executable_create_alt(c.HSA_PROFILE_FULL, c.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, null, &manager.executable);
        if (status != c.HSA_STATUS_SUCCESS) {
            return HsaError.CodeObjectLoadFailed;
        }

        // Load code object
        status = c.hsa_executable_load_agent_code_object(manager.executable, ctx.agent, manager.code_object, null, null);
        if (status != c.HSA_STATUS_SUCCESS) {
            return HsaError.CodeObjectLoadFailed;
        }

        // Freeze executable
        status = c.hsa_executable_freeze(manager.executable, null);
        if (status != c.HSA_STATUS_SUCCESS) {
            return HsaError.CodeObjectLoadFailed;
        }

        return manager;
    }

    fn getKernelSymbol(self: *Self, kernel_name: []const u8) !c.hsa_executable_symbol_t {
        var symbol: c.hsa_executable_symbol_t = undefined;

        const status = c.hsa_executable_get_symbol_by_name(self.executable, kernel_name.ptr, null, &symbol);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Kernel '{}' not found\n", .{kernel_name});
            return HsaError.KernelNotFound;
        }

        return symbol;
    }

    fn deinit(self: *Self) void {
        _ = c.hsa_executable_destroy(self.executable);
        _ = c.hsa_code_object_destroy(self.code_object);
    }
};

// Kernel execution
fn executeKernel(
    ctx: *GpuContext,
    symbol: c.hsa_executable_symbol_t,
    kernargs: []const u8,
    grid_size: [3]u32,
    workgroup_size: [3]u32,
) !void {
    var kernel_object: u64 = undefined;
    var status = c.hsa_executable_symbol_get_info(symbol, c.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
    if (status != c.HSA_STATUS_SUCCESS) {
        return HsaError.ExecutionFailed;
    }

    // Create dispatch packet
    const queue = ctx.queue.?;
    const packet_id = c.hsa_queue_add_write_index_relaxed(queue, 1);
    const packet_ptr = @as(*c.hsa_kernel_dispatch_packet_t, @ptrCast(&queue.base_address[@mod(packet_id, queue.size) * @sizeOf(c.hsa_kernel_dispatch_packet_t)]));

    packet_ptr.header = @as(u16, c.HSA_PACKET_TYPE_KERNEL_DISPATCH) << c.HSA_PACKET_HEADER_TYPE |
        @as(u16, c.HSA_FENCE_SCOPE_SYSTEM) << c.HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
        @as(u16, c.HSA_FENCE_SCOPE_SYSTEM) << c.HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    packet_ptr.setup = 1; // Dimensions
    packet_ptr.workgroup_size_x = @intCast(workgroup_size[0]);
    packet_ptr.workgroup_size_y = @intCast(workgroup_size[1]);
    packet_ptr.workgroup_size_z = @intCast(workgroup_size[2]);
    packet_ptr.grid_size_x = @intCast(grid_size[0]);
    packet_ptr.grid_size_y = @intCast(grid_size[1]);
    packet_ptr.grid_size_z = @intCast(grid_size[2]);
    packet_ptr.kernel_object = kernel_object;
    packet_ptr.kernarg_address = @intFromPtr(kernargs.ptr);
    packet_ptr.private_segment_size = 0;
    packet_ptr.group_segment_size = 2048; // Shared memory size

    // Ring doorbell
    c.hsa_queue_store_write_index_relaxed(queue, packet_id + 1);
    c.hsa_signal_store_relaxed(queue.doorbell_signal, packet_id);

    // Wait for completion
    while (c.hsa_signal_wait_scacquire(queue.doorbell_signal, c.HSA_SIGNAL_CONDITION_LT, packet_id + 1, std.math.maxInt(u64), c.HSA_WAIT_STATE_BLOCKED) != 0) {}
}

// Helper functions for HSA
fn findGpuAgent(agent: c.hsa_agent_t, data: ?*anyopaque) callconv(.C) c.hsa_status_t {
    var device_type: c.hsa_device_type_t = undefined;
    var status = c.hsa_agent_get_info(agent, c.HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != c.HSA_STATUS_SUCCESS) {
        return status;
    }

    if (device_type == c.HSA_DEVICE_TYPE_GPU) {
        const agent_ptr: *c.hsa_agent_t = @ptrCast(@alignCast(data));
        agent_ptr.* = agent;
        return c.HSA_STATUS_INFO_BREAK;
    }

    return c.HSA_STATUS_SUCCESS;
}

fn findMemoryRegions(region: c.hsa_region_t, data: ?*anyopaque) callconv(.C) c.hsa_status_t {
    const ctx: *GpuContext = @ptrCast(@alignCast(data));

    var segment: c.hsa_region_segment_t = undefined;
    var status = c.hsa_region_get_info(region, c.HSA_REGION_INFO_SEGMENT, &segment);
    if (status != c.HSA_STATUS_SUCCESS) {
        return status;
    }

    if (segment == c.HSA_REGION_SEGMENT_KERNARG) {
        ctx.kernarg_region = region;
    } else if (segment == c.HSA_REGION_SEGMENT_GLOBAL) {
        var flags: c.hsa_region_global_flag_t = undefined;
        status = c.hsa_region_get_info(region, c.HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
        if (status != c.HSA_STATUS_SUCCESS) {
            return status;
        }

        if ((flags & c.HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
            ctx.fine_grained_region = region;
        } else {
            ctx.coarse_grained_region = region;
        }
    }

    return c.HSA_STATUS_SUCCESS;
}

// Main application
pub fn main() !void {
    print("Initializing AMD GPU compute with HSA runtime...\n");

    var ctx = GpuContext.init(std.heap.page_allocator) catch |err| {
        print("Failed to initialize GPU context: {}\n", .{err});
        return;
    };
    defer ctx.deinit();

    print("GPU context initialized successfully\n");

    // Load kernels
    var kernel_manager = KernelManager.init(&ctx, "kernel.o") catch |err| {
        print("Failed to load kernels: {}\n", .{err});
        return;
    };
    defer kernel_manager.deinit();

    print("Kernels loaded successfully\n");

    // Test vector addition with shared memory
    try testVectorAddition(&ctx, &kernel_manager);

    // Test matrix multiplication with shared memory
    try testMatrixMultiplication(&ctx, &kernel_manager);

    print("All tests completed successfully!\n");
}

fn testVectorAddition(ctx: *GpuContext, kernel_manager: *KernelManager) !void {
    print("\n=== Testing Vector Addition with Shared Memory ===\n");

    const n: u32 = 1024;
    const workgroup_size: u32 = 256;
    const num_groups = (n + workgroup_size - 1) / workgroup_size;

    // Allocate device memory
    const a = try ctx.allocateMemory(n * @sizeOf(f32), f32);
    const b = try ctx.allocateMemory(n * @sizeOf(f32), f32);
    const c = try ctx.allocateMemory(num_groups * @sizeOf(f32), f32);
    defer ctx.freeMemory(a);
    defer ctx.freeMemory(b);
    defer ctx.freeMemory(c);

    // Initialize input data
    for (0..n) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i * 2);
    }

    // Prepare kernel arguments
    const kernargs = try ctx.allocateKernargs(4 * 8); // 4 pointers
    defer ctx.freeMemory(kernargs);

    const arg_ptrs = std.mem.bytesAsSlice(u64, kernargs);
    arg_ptrs[0] = @intFromPtr(a.ptr);
    arg_ptrs[1] = @intFromPtr(b.ptr);
    arg_ptrs[2] = @intFromPtr(c.ptr);
    arg_ptrs[3] = n;

    // Get kernel symbol and execute
    const symbol = try kernel_manager.getKernelSymbol("vector_add_shared");
    try executeKernel(ctx, symbol, kernargs, .{ num_groups * workgroup_size, 1, 1 }, .{ workgroup_size, 1, 1 });

    // Verify results
    print("First few results: ");
    for (0..@min(8, num_groups)) |i| {
        print("{:.1} ", .{c[i]});
    }
    print("\n");
}

fn testMatrixMultiplication(ctx: *GpuContext, kernel_manager: *KernelManager) !void {
    print("\n=== Testing Matrix Multiplication with Shared Memory ===\n");

    const width: u32 = 128;
    const tile_size: u32 = 16;

    // Allocate device memory
    const a = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    const b = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    const c = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    defer ctx.freeMemory(a);
    defer ctx.freeMemory(b);
    defer ctx.freeMemory(c);

    // Initialize matrices
    for (0..width) |i| {
        for (0..width) |j| {
            a[i * width + j] = @floatFromInt(i + j);
            b[i * width + j] = @floatFromInt(i * j + 1);
            c[i * width + j] = 0.0;
        }
    }

    // Prepare kernel arguments
    const kernargs = try ctx.allocateKernargs(4 * 8);
    defer ctx.freeMemory(kernargs);

    const arg_ptrs = std.mem.bytesAsSlice(u64, kernargs);
    arg_ptrs[0] = @intFromPtr(a.ptr);
    arg_ptrs[1] = @intFromPtr(b.ptr);
    arg_ptrs[2] = @intFromPtr(c.ptr);
    arg_ptrs[3] = width;

    // Execute kernel
    const symbol = try kernel_manager.getKernelSymbol("matrix_multiply_shared");
    const grid_dim = (width + tile_size - 1) / tile_size;

    try executeKernel(ctx, symbol, kernargs, .{ grid_dim * tile_size, grid_dim * tile_size, 1 }, .{ tile_size, tile_size, 1 });

    // Verify a few results
    print("Sample results from C matrix:\n");
    for (0..4) |i| {
        for (0..4) |j| {
            print("{:.1} ", .{c[i * width + j]});
        }
        print("\n");
    }
}
