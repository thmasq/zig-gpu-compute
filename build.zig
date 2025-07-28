const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // GPU target specification
    const amdgcn_mcpu = b.option([]const u8, "gpu", "Target GPU features") orelse "gfx1031";
    const amdgcn_target = b.resolveTargetQuery(std.Build.parseTargetQuery(.{
        .arch_os_abi = "amdgcn-amdhsa-none",
        .cpu_features = amdgcn_mcpu,
    }) catch unreachable);

    // Build GPU kernel library
    const gpu_kernel = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "gpu-kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernel.zig"),
            .target = amdgcn_target,
            .optimize = .ReleaseFast,
        }),
    });
    gpu_kernel.linker_allow_shlib_undefined = false;
    gpu_kernel.bundle_compiler_rt = false;

    // Get the compiled kernel as a binary module
    const kernel_binary = gpu_kernel.getEmittedBin();

    // Host executable
    const exe = b.addExecutable(.{
        .name = "gpu_compute",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    // Link HSA runtime - check multiple possible paths
    const rocm_paths = [_][]const u8{
        "/opt/rocm/lib",
        "/usr/lib/x86_64-linux-gnu", // Ubuntu/Debian
        "/usr/lib64", // RHEL/CentOS
    };

    const rocm_include_paths = [_][]const u8{
        "/opt/rocm/include",
        "/usr/include/hsa",
        "/usr/include",
    };

    // Try to find ROCm installation
    var rocm_lib_found = false;
    var rocm_include_found = false;

    for (rocm_paths) |path| {
        if (std.fs.cwd().access(path, .{})) {
            exe.addLibraryPath(.{ .cwd_relative = path });
            rocm_lib_found = true;
            break;
        } else |_| {}
    }

    for (rocm_include_paths) |path| {
        if (std.fs.cwd().access(path, .{})) {
            exe.addIncludePath(.{ .cwd_relative = path });
            rocm_include_found = true;
            break;
        } else |_| {}
    }

    if (!rocm_lib_found) {
        std.log.warn("ROCm library path not found, you may need to set LD_LIBRARY_PATH\n", .{});
    }

    if (!rocm_include_found) {
        std.log.warn("ROCm include path not found, compilation may fail\n", .{});
    }

    exe.linkSystemLibrary("hsa-runtime64");

    // Embed the GPU kernel binary in the host application
    exe.root_module.addAnonymousImport("gpu-kernel", .{
        .root_source_file = kernel_binary,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Optional: Add a step to just build the kernel
    const kernel_step = b.step("kernel", "Build GPU kernel only");
    kernel_step.dependOn(&gpu_kernel.step);
}
