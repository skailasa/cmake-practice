
template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 16, 16, 16, SM< 700 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 32, 32, 32, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 48, 48, 48, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 64, 64, 64, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 80, 80, 80, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 96, 96, 96, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 112, 112, 112, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 128, 128, 128, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 128, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 64, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 144, 144, 144, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 160, 160, 160, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 32, 2>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 176, 176, 176, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 192, 192, 192, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 32, 2>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 208, 208, 208, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 221, 221, 221, SM< 700 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 32, 2>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};
