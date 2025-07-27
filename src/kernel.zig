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
    // Get work item and group IDs using external intrinsics
    const gid = @"llvm.amdgcn.workitem.id.x"();
    const lid = @"llvm.amdgcn.workgroup.id.x"();

    // For workgroup size, we'll use a constant since it's set at dispatch
    const lsize: u32 = 256; // This should match the workgroup size in dispatch

    const global_id = lid * lsize + gid;
    const local_id = gid;

    if (global_id >= n) return;

    // Load data into shared memory
    shared_mem[local_id] = a[global_id] + b[global_id];

    // Synchronize workgroup
    @"llvm.amdgcn.s.barrier"();

    // Perform reduction in shared memory
    var stride: u32 = lsize / 2;
    while (stride > 0) : (stride /= 2) {
        if (local_id < stride) {
            shared_mem[local_id] += shared_mem[local_id + stride];
        }
        @"llvm.amdgcn.s.barrier"();
    }

    // Write result
    if (local_id == 0) {
        c[lid] = shared_mem[0];
    } else if (global_id < n) {
        c[global_id] = shared_mem[local_id];
    }
}

export fn matrix_multiply_shared(a: [*]addrspace(.global) const f32, b: [*]addrspace(.global) const f32, c: [*]addrspace(.global) f32, width: u32) callconv(.Kernel) void {
    const tx = @"llvm.amdgcn.workitem.id.x"();
    const ty = @"llvm.amdgcn.workitem.id.y"();
    const bx = @"llvm.amdgcn.workgroup.id.x"();
    const by = @"llvm.amdgcn.workgroup.id.y"();

    const TILE_SIZE = 16;

    // Use global shared memory instead of local variables
    const as_tile = @as(*[TILE_SIZE][TILE_SIZE]f32, @ptrFromInt(@intFromPtr(&shared_mem[0])));
    const bs_tile = @as(*[TILE_SIZE][TILE_SIZE]f32, @ptrFromInt(@intFromPtr(&shared_mem[256])));

    const row = by * TILE_SIZE + ty;
    const col = bx * TILE_SIZE + tx;

    var c_val: f32 = 0.0;

    // Tile across the K dimension
    var tile: u32 = 0;
    while (tile < (width + TILE_SIZE - 1) / TILE_SIZE) : (tile += 1) {
        // Load tiles into shared memory
        const a_col = tile * TILE_SIZE + tx;
        const b_row = tile * TILE_SIZE + ty;

        if (row < width and a_col < width) {
            as_tile[ty][tx] = a[row * width + a_col];
        } else {
            as_tile[ty][tx] = 0.0;
        }

        if (b_row < width and col < width) {
            bs_tile[ty][tx] = b[b_row * width + col];
        } else {
            bs_tile[ty][tx] = 0.0;
        }

        @"llvm.amdgcn.s.barrier"();

        // Compute partial result
        var k: u32 = 0;
        while (k < TILE_SIZE) : (k += 1) {
            c_val += as_tile[ty][k] * bs_tile[k][tx];
        }

        @"llvm.amdgcn.s.barrier"();
    }

    // Write result
    if (row < width and col < width) {
        c[row * width + col] = c_val;
    }
}
