const std = @import("std");
const builtin = @import("builtin");

// AMDGCN intrinsics as external functions
extern fn @"llvm.amdgcn.workitem.id.x"() u32;
extern fn @"llvm.amdgcn.workitem.id.y"() u32;
extern fn @"llvm.amdgcn.workgroup.id.x"() u32;
extern fn @"llvm.amdgcn.workgroup.id.y"() u32;
extern fn @"llvm.amdgcn.s.barrier"() void;

// Shared memory declaration using addrspace
var shared_mem: [512]f32 addrspace(.shared) = undefined;

// Kernel attributes for AMD GPU
export fn vector_add_shared(a: [*]addrspace(.global) const f32, b: [*]addrspace(.global) const f32, c: [*]addrspace(.global) f32, n: u32) callconv(.Kernel) void {
    // Correct ID usage
    const local_id = @"llvm.amdgcn.workitem.id.x"(); // 0-255 within workgroup
    const group_id = @"llvm.amdgcn.workgroup.id.x"(); // Workgroup index

    const lsize: u32 = 256; // Workgroup size
    const global_id = group_id * lsize + local_id;

    if (global_id >= n) return;

    // Load data into shared memory
    shared_mem[local_id] = a[global_id] + b[global_id];

    // Synchronize workgroup
    @"llvm.amdgcn.s.barrier"();

    // Perform reduction in shared memory
    var stride: u32 = lsize / 2;
    while (stride > 0) : (stride /= 2) {
        if (local_id < stride and local_id + stride < lsize) {
            shared_mem[local_id] += shared_mem[local_id + stride];
        }
        @"llvm.amdgcn.s.barrier"();
    }

    // Only thread 0 writes the reduced result for this workgroup
    if (local_id == 0) {
        c[group_id] = shared_mem[0]; // One result per workgroup
    }
}

export fn matrix_multiply_shared(a: [*]addrspace(.global) const f32, b: [*]addrspace(.global) const f32, c: [*]addrspace(.global) f32, width: u32) callconv(.Kernel) void {
    // Correct ID usage
    const local_x = @"llvm.amdgcn.workitem.id.x"(); // Local thread ID in X
    const local_y = @"llvm.amdgcn.workitem.id.y"(); // Local thread ID in Y
    const group_x = @"llvm.amdgcn.workgroup.id.x"(); // Workgroup ID in X
    const group_y = @"llvm.amdgcn.workgroup.id.y"(); // Workgroup ID in Y

    const TILE_SIZE = 16;

    // Use global shared memory instead of local variables
    const as_tile = @as(*[TILE_SIZE][TILE_SIZE]f32, @ptrFromInt(@intFromPtr(&shared_mem[0])));
    const bs_tile = @as(*[TILE_SIZE][TILE_SIZE]f32, @ptrFromInt(@intFromPtr(&shared_mem[256])));

    const row = group_y * TILE_SIZE + local_y;
    const col = group_x * TILE_SIZE + local_x;

    var c_val: f32 = 0.0;

    // Tile across the K dimension
    var tile: u32 = 0;
    while (tile < (width + TILE_SIZE - 1) / TILE_SIZE) : (tile += 1) {
        // Load tiles into shared memory
        const a_col = tile * TILE_SIZE + local_x;
        const b_row = tile * TILE_SIZE + local_y;

        if (row < width and a_col < width) {
            as_tile[local_y][local_x] = a[row * width + a_col];
        } else {
            as_tile[local_y][local_x] = 0.0;
        }

        if (b_row < width and col < width) {
            bs_tile[local_y][local_x] = b[b_row * width + col];
        } else {
            bs_tile[local_y][local_x] = 0.0;
        }

        @"llvm.amdgcn.s.barrier"();

        // Compute partial result
        var k: u32 = 0;
        while (k < TILE_SIZE) : (k += 1) {
            c_val += as_tile[local_y][k] * bs_tile[k][local_x];
        }

        @"llvm.amdgcn.s.barrier"();
    }

    // Write result
    if (row < width and col < width) {
        c[row * width + col] = c_val;
    }
}
