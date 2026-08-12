target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    add_rules("cuda")
    add_syslinks("cudart")
    set_policy("build.cuda.devlink", true)
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
    else
        add_cuflags("-Xcompiler=/MD")
    end

    add_cuflags("--expt-relaxed-constexpr")
    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("cuda")
    add_syslinks("cudart")
    set_policy("build.cuda.devlink", true)
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
    else
        add_cuflags("-Xcompiler=/MD")
    end

    add_cuflags("--expt-relaxed-constexpr")
    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
