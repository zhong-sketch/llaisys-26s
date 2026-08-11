target("llaisys-device-iluvatar")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_includedirs("/usr/local/corex-4.4.0/include",
                        "/usr/local/cuda-10.2/include",
                        {public = true})
        add_linkdirs("/usr/local/corex-4.4.0/lib64",
                     "/usr/local/cuda-10.2/lib64",
                     {public = true})
        add_links("cudart", {public = true})
    end

    add_files("../src/device/iluvatar/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_includedirs("/usr/local/corex-4.4.0/include",
                        "/usr/local/cuda-10.2/include",
                        {public = true})
        add_linkdirs("/usr/local/corex-4.4.0/lib64",
                     "/usr/local/cuda-10.2/lib64",
                     {public = true})
        add_links("cudart", {public = true})
    end

    add_files("../src/ops/*/iluvatar/*.cpp")

    on_install(function (target) end)
target_end()
