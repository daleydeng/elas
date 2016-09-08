env = DefaultEnvironment()
# SSE4_1 will cause simdpp::abs failed
env.Append(
    CXXFLAGS=["-O2", "-Wall", "-std=c++11", "-fdiagnostics-color=auto", '-DSIMDPP_ARCH_X86_SSE3'],
    # CXXFLAGS=["-O2", "-Wall", "-std=c++11", "-fdiagnostics-color=auto", '-DSIMDPP_ARCH_ARM_NEON'],
    LINKFLAGS="-Wl,--unresolved-symbols=ignore-in-shared-libs -Wl,--as-needed",
    CPPPATH=['/usr/local/include', '../include', '/usr/include/eigen3', '.'],
    LIBPATH=['/usr/local/lib', '/usr/local/lib64'],
    LIBS=['glog', 'CGAL', 'gmp']
)

Program("app/elas", Glob('*.cc')+Glob('app/*.cc'))
