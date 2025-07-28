const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;

// Import the embedded GPU kernel
const gpu_kernel_binary = @embedFile("gpu-kernel");

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
        var status = c.hsa_memory_allocate(self.coarse_grained_region.?, size, &ptr);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Memory allocation failed with status: {} (0x{X})\n", .{ status, status });
            return HsaError.MemoryAllocationFailed;
        }

        print("Granting GPU agent access to allocated memory...\n", .{});
        status = c.hsa_amd_agents_allow_access(1, &self.agent, null, ptr);
        if (status != c.HSA_STATUS_SUCCESS) {
            print("Failed to grant GPU access to memory: {} (0x{X})\n", .{ status, status });
            _ = c.hsa_memory_free(ptr);
            return HsaError.MemoryAllocationFailed;
        }
        print("GPU access granted successfully\n", .{});

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
    agent: c.hsa_agent_t,

    const Self = @This();

    fn init(ctx: *GpuContext) !Self {
        print("Loading embedded kernel binary ({} bytes)...\n", .{gpu_kernel_binary.len});

        if (gpu_kernel_binary.len >= 4) {
            print("Binary header: 0x{X:0>2} 0x{X:0>2} 0x{X:0>2} 0x{X:0>2}\n", .{ gpu_kernel_binary[0], gpu_kernel_binary[1], gpu_kernel_binary[2], gpu_kernel_binary[3] });
        }

        var manager = Self{
            .code_object_reader = undefined,
            .executable = undefined,
            .agent = ctx.agent,
        };

        // Create code object reader from embedded binary
        print("Creating code object reader from embedded binary...\n", .{});
        var status = c.hsa_code_object_reader_create_from_memory(gpu_kernel_binary.ptr, gpu_kernel_binary.len, &manager.code_object_reader);
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
    var status = c.hsa_executable_symbol_get_info(symbol, c.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
    if (status != c.HSA_STATUS_SUCCESS) {
        print("Failed to get kernel object: {} (0x{X})\n", .{ status, status });
        return HsaError.ExecutionFailed;
    }

    // Determine number of dimensions
    var num_dims: u32 = 1;
    if (grid_size[2] > 1 or workgroup_size[2] > 1) {
        num_dims = 3;
    } else if (grid_size[1] > 1 or workgroup_size[1] > 1) {
        num_dims = 2;
    }

    print("Executing kernel with {} dimensions\n", .{num_dims});
    print("Grid [{}, {}, {}] workgroup [{}, {}, {}]\n", .{ grid_size[0], grid_size[1], grid_size[2], workgroup_size[0], workgroup_size[1], workgroup_size[2] });

    const queue = ctx.queue.?;

    // Create a completion signal
    var completion_signal: c.hsa_signal_t = undefined;
    status = c.hsa_signal_create(1, 0, null, &completion_signal);
    if (status != c.HSA_STATUS_SUCCESS) {
        print("Failed to create completion signal: {} (0x{X})\n", .{ status, status });
        return HsaError.ExecutionFailed;
    }
    defer _ = c.hsa_signal_destroy(completion_signal);

    const packet_id = c.hsa_queue_add_write_index_relaxed(queue, 1);
    const base_packets = @as([*]c.hsa_kernel_dispatch_packet_t, @ptrCast(@alignCast(queue.base_address)));
    const packet_ptr = &base_packets[@mod(packet_id, queue.size)];

    // Clear the packet first
    @memset(@as([*]u8, @ptrCast(packet_ptr))[0..@sizeOf(c.hsa_kernel_dispatch_packet_t)], 0);

    packet_ptr.setup = @as(u16, @intCast(num_dims)) << c.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

    packet_ptr.header = @as(u16, c.HSA_PACKET_TYPE_KERNEL_DISPATCH) << c.HSA_PACKET_HEADER_TYPE |
        @as(u16, c.HSA_FENCE_SCOPE_SYSTEM) << c.HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
        @as(u16, c.HSA_FENCE_SCOPE_SYSTEM) << c.HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    packet_ptr.workgroup_size_x = @intCast(workgroup_size[0]);
    packet_ptr.workgroup_size_y = @intCast(workgroup_size[1]);
    packet_ptr.workgroup_size_z = @intCast(workgroup_size[2]);
    packet_ptr.grid_size_x = @intCast(grid_size[0]);
    packet_ptr.grid_size_y = @intCast(grid_size[1]);
    packet_ptr.grid_size_z = @intCast(grid_size[2]);
    packet_ptr.kernel_object = kernel_object;
    packet_ptr.kernarg_address = @ptrCast(@constCast(kernargs.ptr));
    packet_ptr.private_segment_size = 0;
    packet_ptr.group_segment_size = 2048; // Shared memory size
    packet_ptr.completion_signal = completion_signal;

    print("Packet setup: dimensions={}, setup=0x{X}\n", .{ num_dims, packet_ptr.setup });

    // Submit the packet
    c.hsa_queue_store_write_index_relaxed(queue, packet_id + 1);
    c.hsa_signal_store_relaxed(queue.doorbell_signal, @intCast(packet_id));

    // Wait for completion using the completion signal
    print("Waiting for kernel completion...\n", .{});
    const wait_result = c.hsa_signal_wait_scacquire(completion_signal, c.HSA_SIGNAL_CONDITION_EQ, 0, std.math.maxInt(u64), c.HSA_WAIT_STATE_BLOCKED);

    if (wait_result != 0) {
        print("Kernel execution may have failed or timed out\n", .{});
        return HsaError.ExecutionFailed;
    }

    print("Kernel completed successfully\n", .{});
}

fn matrixMultiplyCPU(a: []const f32, b: []const f32, result: []f32, width: u32) void {
    // Initialize result matrix to zero
    for (result) |*element| {
        element.* = 0.0;
    }

    // Standard matrix multiplication: C[i][j] = sum(A[i][k] * B[k][j])
    for (0..width) |i| {
        for (0..width) |j| {
            var sum: f32 = 0.0;
            for (0..width) |k| {
                sum += a[i * width + k] * b[k * width + j];
            }
            result[i * width + j] = sum;
        }
    }
}

// Function to compare two matrices with tolerance
fn compareMatrices(gpu_result: []const f32, cpu_result: []const f32, width: u32, tolerance: f32) bool {
    var max_diff: f32 = 0.0;
    var num_errors: u32 = 0;

    for (0..width) |i| {
        for (0..width) |j| {
            const idx = i * width + j;
            const diff = @abs(gpu_result[idx] - cpu_result[idx]);
            max_diff = @max(max_diff, diff);

            if (diff > tolerance) {
                if (num_errors < 10) { // Only print first 10 errors
                    print("Mismatch at [{}, {}]: GPU={:.6}, CPU={:.6}, diff={:.6}\n", .{ i, j, gpu_result[idx], cpu_result[idx], diff });
                }
                num_errors += 1;
            }
        }
    }

    print("Maximum difference: {:.8}\n", .{max_diff});
    if (num_errors > 0) {
        print("Total errors with tolerance {:.6}: {}\n", .{ tolerance, num_errors });
        return false;
    }

    return true;
}

pub fn main() !void {
    print("Initializing AMD GPU compute with HSA runtime...\n", .{});

    var ctx = GpuContext.init() catch |err| {
        print("Failed to initialize GPU context: {}\n", .{err});
        return;
    };
    defer ctx.deinit();

    print("GPU context initialized successfully\n", .{});

    var kernel_manager = KernelManager.init(&ctx) catch |err| {
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
    const result_c = try ctx.allocateMemory(n * @sizeOf(f32), f32);
    defer ctx.freeMemory(a);
    defer ctx.freeMemory(b);
    defer ctx.freeMemory(result_c);

    for (0..n) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i * 2);
    }

    const kernargs = try ctx.allocateKernargs(32);
    defer ctx.freeMemory(kernargs);

    const arg_ptrs = std.mem.bytesAsSlice(u64, kernargs);
    arg_ptrs[0] = @intFromPtr(a.ptr);
    arg_ptrs[1] = @intFromPtr(b.ptr);
    arg_ptrs[2] = @intFromPtr(result_c.ptr);
    // Write the u32 value properly
    const n_ptr = @as(*u32, @ptrCast(@alignCast(&kernargs[24])));
    n_ptr.* = n;

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

    // Allocate GPU memory
    const a = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    const b = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    const gpu_result = try ctx.allocateMemory(width * width * @sizeOf(f32), f32);
    defer ctx.freeMemory(a);
    defer ctx.freeMemory(b);
    defer ctx.freeMemory(gpu_result);

    // Allocate CPU memory for comparison
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cpu_result = try allocator.alloc(f32, width * width);
    defer allocator.free(cpu_result);

    // Initialize input matrices
    print("Initializing matrices...\n", .{});
    for (0..width) |i| {
        for (0..width) |j| {
            a[i * width + j] = @floatFromInt(i + j);
            b[i * width + j] = @floatFromInt(i * j + 1);
            gpu_result[i * width + j] = 0.0;
        }
    }

    // Setup kernel arguments
    const kernargs = try ctx.allocateKernargs(28);
    defer ctx.freeMemory(kernargs);

    const arg_ptrs = std.mem.bytesAsSlice(u64, kernargs[0..24]);
    arg_ptrs[0] = @intFromPtr(a.ptr);
    arg_ptrs[1] = @intFromPtr(b.ptr);
    arg_ptrs[2] = @intFromPtr(gpu_result.ptr);

    const width_ptr = @as(*u32, @ptrCast(@alignCast(&kernargs[24])));
    width_ptr.* = width;

    // Execute GPU kernel
    print("Executing GPU matrix multiplication...\n", .{});
    const start_gpu = std.time.nanoTimestamp();

    const symbol = try kernel_manager.getKernelSymbol("matrix_multiply_shared");
    const grid_dim = (width + tile_size - 1) / tile_size;

    try executeKernel(ctx, symbol, kernargs, .{ grid_dim * tile_size, grid_dim * tile_size, 1 }, .{ tile_size, tile_size, 1 });

    const end_gpu = std.time.nanoTimestamp();
    const gpu_time_ms = @as(f64, @floatFromInt(end_gpu - start_gpu)) / 1_000_000.0;

    // Execute CPU matrix multiplication
    print("Executing CPU matrix multiplication...\n", .{});
    const start_cpu = std.time.nanoTimestamp();

    matrixMultiplyCPU(a, b, cpu_result, width);

    const end_cpu = std.time.nanoTimestamp();
    const cpu_time_ms = @as(f64, @floatFromInt(end_cpu - start_cpu)) / 1_000_000.0;

    // Performance comparison
    print("\n=== Performance Results ===\n", .{});
    print("GPU time: {:.2} ms\n", .{gpu_time_ms});
    print("CPU time: {:.2} ms\n", .{cpu_time_ms});
    print("Speedup: {:.2}x\n", .{cpu_time_ms / gpu_time_ms});

    // Display sample results
    print("\n=== Sample Results ===\n", .{});
    print("GPU results (top-left 4x4):\n", .{});
    for (0..4) |i| {
        for (0..4) |j| {
            print("{:.1} ", .{gpu_result[i * width + j]});
        }
        print("\n", .{});
    }

    print("CPU results (top-left 4x4):\n", .{});
    for (0..4) |i| {
        for (0..4) |j| {
            print("{:.1} ", .{cpu_result[i * width + j]});
        }
        print("\n", .{});
    }

    // Compare results
    print("\n=== Correctness Verification ===\n", .{});
    const tolerance: f32 = 1e-5;
    const results_match = compareMatrices(gpu_result, cpu_result, width, tolerance);

    if (results_match) {
        print("✅ SUCCESS: GPU and CPU results match within tolerance ({:.6})\n", .{tolerance});
    } else {
        print("❌ FAILURE: GPU and CPU results do not match!\n", .{});
        return error.ResultMismatch;
    }

    // Additional statistics
    var sum_gpu: f64 = 0.0;
    var sum_cpu: f64 = 0.0;
    for (0..width * width) |i| {
        sum_gpu += gpu_result[i];
        sum_cpu += cpu_result[i];
    }

    print("Matrix sum - GPU: {:.6}, CPU: {:.6}\n", .{ sum_gpu, sum_cpu });
    print("Matrix multiplication test completed successfully!\n", .{});
}
