env = DefaultEnvironment()
env.Append(
    CXXFLAGS=["-O2", "-Wall", "-std=c++11", "-fdiagnostics-color=auto", '-DSIMDPP_ARCH_X86_SSE4_1'],
    LINKFLAGS="-Wl,--unresolved-symbols=ignore-in-shared-libs -Wl,--as-needed",
    CPPPATH=['/usr/local/include', '../include'],
    LIBPATH=['/usr/local/lib', '/usr/local/lib64'],
    LIBS=['glog']
)

Program("elas", Glob("*.cpp")+Glob('*.cc'))
