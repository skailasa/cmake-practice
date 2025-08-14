//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<33, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<33,   33,   33,    1,    32,     0,   511>
    >;
};

template<> struct block_fft_record<34, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,   34,   34,    1,    32,     0,   512>
    >;
};

template<> struct block_fft_record<35, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<35,   35,   35,    1,    32,     0,   513>
    >;
};

template<> struct block_fft_record<37, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<37,   37,   37,    1,    32,     0,   514>
    >;
};

template<> struct block_fft_record<38, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,   38,   38,    1,    32,     0,   515>
    >;
};

template<> struct block_fft_record<39, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<39,   39,   39,    1,    32,     0,   516>
    >;
};

template<> struct block_fft_record<40, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,    2,   10,    4,     8,   320,   519>,
        block_fft_implementation<10,    2,   10,    4,     8,   640,   520>,
        block_fft_implementation<20,    2,   20,    2,    16,   640,   518>,
        block_fft_implementation<20,    2,   20,    2,    16,   320,   517>,
        block_fft_implementation<40,   40,   40,    1,    32,     0,   521>
    >;
};

template<> struct block_fft_record<2, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    2,    1,    32,     0,   522>
    >;
};

template<> struct block_fft_record<4, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    4,    1,    32,     0,   523>,
        block_fft_implementation< 2,    2,    2,    2,    16,    32,   524>,
        block_fft_implementation< 2,    2,    2,    2,    16,    64,   525>
    >;
};

template<> struct block_fft_record<8, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    1,    32,     0,   526>,
        block_fft_implementation< 4,    2,    4,    2,    16,    64,   527>,
        block_fft_implementation< 2,    2,    2,    4,     8,    64,   528>,
        block_fft_implementation< 2,    2,    2,    4,     8,   128,   530>,
        block_fft_implementation< 4,    2,    4,    2,    16,   128,   529>
    >;
};

template<> struct block_fft_record<16, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,    1,    32,     0,   531>,
        block_fft_implementation< 4,    4,    4,    4,     8,   128,   533>,
        block_fft_implementation< 4,    4,    4,    4,     8,   256,   532>,
        block_fft_implementation< 2,    2,    2,    8,     4,   128,   537>,
        block_fft_implementation< 2,    2,    2,    8,     4,   256,   534>,
        block_fft_implementation< 8,    2,    8,    2,    16,   128,   535>,
        block_fft_implementation< 8,    2,    8,    2,    16,   256,   536>
    >;
};

template<> struct block_fft_record<32, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<32,   32,   32,    1,    32,     0,   538>,
        block_fft_implementation< 8,    4,    8,    4,     8,   256,   539>,
        block_fft_implementation< 4,    2,    4,    8,     4,   256,   542>,
        block_fft_implementation< 2,    2,    2,   16,     2,   256,   545>,
        block_fft_implementation< 8,    4,    8,    4,     8,   512,   541>,
        block_fft_implementation< 2,    2,    2,   16,     2,   512,   544>,
        block_fft_implementation< 4,    2,    4,    8,     4,   512,   540>,
        block_fft_implementation<16,    2,   16,    2,    16,   256,   546>,
        block_fft_implementation<16,    2,   16,    2,    16,   512,   543>
    >;
};

template<> struct block_fft_record<64, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    8,     4,   512,   547>,
        block_fft_implementation< 4,    4,    4,   16,     2,   512,   548>,
        block_fft_implementation< 4,    4,    4,   16,     2,  1024,   549>,
        block_fft_implementation< 8,    8,    8,    8,     4,  1024,   550>,
        block_fft_implementation<16,    4,   16,    4,     8,   512,   552>,
        block_fft_implementation< 2,    2,    2,   32,     1,   512,   554>,
        block_fft_implementation<16,    4,   16,    4,     8,  1024,   551>,
        block_fft_implementation< 2,    2,    2,   32,     1,  1024,   553>
    >;
};

template<> struct block_fft_record<128, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    8,   16,    8,     4,  1024,   557>,
        block_fft_implementation<16,    8,   16,    8,     4,  2048,   556>,
        block_fft_implementation< 4,    2,    4,   32,     1,  1024,   559>,
        block_fft_implementation< 2,    2,    2,   64,     1,  1024,   562>,
        block_fft_implementation< 2,    2,    2,   64,     1,  2048,   561>,
        block_fft_implementation< 4,    2,    4,   32,     1,  2048,   558>,
        block_fft_implementation< 8,    2,    8,   16,     2,  1024,   555>,
        block_fft_implementation< 8,    2,    8,   16,     2,  2048,   560>
    >;
};

template<> struct block_fft_record<256, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    4,   64,     1,  2048,   567>,
        block_fft_implementation<16,   16,   16,   16,     2,  2048,   565>,
        block_fft_implementation< 4,    4,    4,   64,     1,  4096,   566>,
        block_fft_implementation<16,   16,   16,   16,     2,  4096,   564>,
        block_fft_implementation< 8,    4,    8,   32,     1,  2048,   563>,
        block_fft_implementation< 2,    2,    2,  128,     1,  2048,   569>,
        block_fft_implementation< 8,    4,    8,   32,     1,  4096,   568>,
        block_fft_implementation< 2,    2,    2,  128,     1,  4096,   570>
    >;
};

template<> struct block_fft_record<512, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,   64,     1,  4096,   571>,
        block_fft_implementation< 8,    8,    8,   64,     1,  8192,   572>,
        block_fft_implementation< 4,    2,    4,  128,     1,  4096,   575>,
        block_fft_implementation< 4,    2,    4,  128,     1,  8192,   576>,
        block_fft_implementation< 2,    2,    2,  256,     1,  4096,   577>,
        block_fft_implementation< 2,    2,    2,  256,     1,  8192,   578>,
        block_fft_implementation<16,    2,   16,   32,     1,  4096,   574>,
        block_fft_implementation<16,    2,   16,   32,     1,  8192,   573>
    >;
};

template<> struct block_fft_record<1024, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    4,  256,     1,  8192,   581>,
        block_fft_implementation< 4,    4,    4,  256,     1, 16384,   582>,
        block_fft_implementation<16,    4,   16,   64,     1,  8192,   580>,
        block_fft_implementation< 2,    2,    2,  512,     1, 16384,   585>,
        block_fft_implementation<16,    4,   16,   64,     1, 16384,   583>,
        block_fft_implementation< 2,    2,    2,  512,     1,  8192,   586>,
        block_fft_implementation< 8,    2,    8,  128,     1,  8192,   579>,
        block_fft_implementation< 8,    2,    8,  128,     1, 16384,   584>
    >;
};

template<> struct block_fft_record<2048, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    8,   16,  128,     1, 16384,   588>,
        block_fft_implementation< 4,    2,    4,  512,     1, 16384,   591>,
        block_fft_implementation< 8,    4,    8,  256,     1, 16384,   587>,
        block_fft_implementation<16,    8,   16,  128,     1, 32768,   589>,
        block_fft_implementation< 4,    2,    4,  512,     1, 32768,   592>,
        block_fft_implementation< 2,    2,    2, 1024,     1, 32768,   594>,
        block_fft_implementation< 8,    4,    8,  256,     1, 32768,   590>,
        block_fft_implementation< 2,    2,    2, 1024,     1, 16384,   593>
    >;
};

template<> struct block_fft_record<4096, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    4, 1024,     1, 32768,   599>,
        block_fft_implementation< 4,    4,    4, 1024,     1, 65536,   600>,
        block_fft_implementation<16,   16,   16,  256,     1, 65536,   597>,
        block_fft_implementation< 8,    8,    8,  512,     1, 32768,   595>,
        block_fft_implementation<16,   16,   16,  256,     1, 32768,   596>,
        block_fft_implementation< 8,    8,    8,  512,     1, 65536,   598>
    >;
};

template<> struct block_fft_record<8192, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    2,    8, 1024,     1, 65536,   602>,
        block_fft_implementation<16,    2,   16,  512,     1, 65536,   601>
    >;
};

template<> struct block_fft_record<3, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    3,    1,    32,     0,   603>
    >;
};

template<> struct block_fft_record<9, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    1,    32,     0,   604>,
        block_fft_implementation< 3,    3,    3,    3,    32,    72,   605>,
        block_fft_implementation< 3,    3,    3,    3,    32,   144,   606>
    >;
};

template<> struct block_fft_record<27, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,    1,    32,     0,   607>,
        block_fft_implementation< 3,    3,    3,    9,     7,   216,   610>,
        block_fft_implementation< 3,    3,    3,    9,     7,   432,   611>,
        block_fft_implementation< 9,    3,    9,    3,    32,   216,   608>,
        block_fft_implementation< 9,    3,    9,    3,    32,   432,   609>
    >;
};

template<> struct block_fft_record<81, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    3,   27,     7,   648,   617>,
        block_fft_implementation< 3,    3,    3,   27,    13,  1296,   616>,
        block_fft_implementation< 9,    9,    9,    9,    32,   648,   613>,
        block_fft_implementation< 9,    9,    9,    9,    32,  1296,   612>,
        block_fft_implementation<27,    3,   27,    3,    32,   648,   614>,
        block_fft_implementation<27,    3,   27,    3,    32,  1296,   615>
    >;
};

template<> struct block_fft_record<243, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    3,   81,    11,  3888,   622>,
        block_fft_implementation< 3,    3,    3,   81,     5,  1944,   623>,
        block_fft_implementation<27,    9,   27,    9,     7,  1944,   618>,
        block_fft_implementation<27,    9,   27,    9,     7,  3888,   619>,
        block_fft_implementation< 9,    3,    9,   27,    13,  1944,   621>,
        block_fft_implementation< 9,    3,    9,   27,    13,  3888,   620>
    >;
};

template<> struct block_fft_record<729, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    3,  243,     3, 11664,   629>,
        block_fft_implementation< 3,    3,    3,  243,     3,  5832,   628>,
        block_fft_implementation<27,   27,   27,   27,     7,  5832,   626>,
        block_fft_implementation< 9,    9,    9,   81,    11,  5832,   625>,
        block_fft_implementation< 9,    9,    9,   81,     5, 11664,   624>,
        block_fft_implementation<27,   27,   27,   27,     2, 11664,   627>
    >;
};

template<> struct block_fft_record<2187, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    3,  729,     1, 34992,   633>,
        block_fft_implementation< 3,    3,    3,  729,     1, 17496,   632>,
        block_fft_implementation< 9,    3,    9,  243,     3, 17496,   630>,
        block_fft_implementation< 9,    3,    9,  243,     1, 34992,   631>
    >;
};

template<> struct block_fft_record<6561, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,  729,     1, 52488,   634>
    >;
};

template<> struct block_fft_record<5, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    5,    1,    32,     0,   635>
    >;
};

template<> struct block_fft_record<25, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,    1,    32,     0,   636>,
        block_fft_implementation< 5,    5,    5,    5,    32,   200,   638>,
        block_fft_implementation< 5,    5,    5,    5,    32,   400,   637>
    >;
};

template<> struct block_fft_record<125, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    5,   25,    14,  1000,   640>,
        block_fft_implementation<25,    5,   25,    5,    19,  1000,   641>,
        block_fft_implementation< 5,    5,    5,   25,    14,  2000,   639>,
        block_fft_implementation<25,    5,   25,    5,    32,  2000,   642>
    >;
};

template<> struct block_fft_record<625, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,   25,     5,  5000,   643>,
        block_fft_implementation<25,   25,   25,   25,     5, 10000,   646>,
        block_fft_implementation< 5,    5,    5,  125,     1,  5000,   644>,
        block_fft_implementation< 5,    5,    5,  125,     1, 10000,   645>
    >;
};

template<> struct block_fft_record<3125, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    5,  625,     1, 25000,   648>,
        block_fft_implementation< 5,    5,    5,  625,     1, 50000,   647>
    >;
};

template<> struct block_fft_record<7, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,    1,    32,     0,   649>
    >;
};

template<> struct block_fft_record<49, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,    7,    32,   392,   650>,
        block_fft_implementation< 7,    7,    7,    7,    32,   784,   651>
    >;
};

template<> struct block_fft_record<343, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,   49,    13,  2744,   653>,
        block_fft_implementation< 7,    7,    7,   49,     9,  5488,   652>
    >;
};

template<> struct block_fft_record<2401, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,  343,     1, 19208,   655>,
        block_fft_implementation< 7,    7,    7,  343,     1, 38416,   654>
    >;
};

template<> struct block_fft_record<11, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    1,    32,     0,   656>
    >;
};

template<> struct block_fft_record<121, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,   11,    29,   968,   658>,
        block_fft_implementation<11,   11,   11,   11,    29,  1936,   657>
    >;
};

template<> struct block_fft_record<1331, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,  121,     5, 10648,   659>,
        block_fft_implementation<11,   11,   11,  121,     1, 21296,   660>
    >;
};

template<> struct block_fft_record<6, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    1,    32,     0,   661>
    >;
};

template<> struct block_fft_record<36, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    6,    16,   288,   663>,
        block_fft_implementation< 6,    6,    6,    6,    16,   576,   662>,
        block_fft_implementation<36,   36,   36,    1,    32,     0,   664>
    >;
};

template<> struct block_fft_record<216, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,   36,     8,  1728,   665>,
        block_fft_implementation< 6,    6,    6,   36,     8,  3456,   666>
    >;
};

template<> struct block_fft_record<1296, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,  216,     4, 10368,   667>,
        block_fft_implementation< 6,    6,    6,  216,     1, 20736,   668>
    >;
};

template<> struct block_fft_record<7776, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<36,    6,   36,  216,     1, 62208,   669>
    >;
};

template<> struct block_fft_record<10, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,    1,    32,     0,   670>
    >;
};

template<> struct block_fft_record<100, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,   10,    16,   800,   672>,
        block_fft_implementation<10,   10,   10,   10,    16,  1600,   671>
    >;
};

template<> struct block_fft_record<1000, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,  100,     6,  8000,   674>,
        block_fft_implementation<10,   10,   10,  100,     3, 16000,   673>
    >;
};

template<> struct block_fft_record<12, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,    1,    32,     0,   676>
    >;
};

template<> struct block_fft_record<144, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,   12,     8,  1152,   677>,
        block_fft_implementation<12,   12,   12,   12,     8,  2304,   678>
    >;
};

template<> struct block_fft_record<1728, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,  144,     2, 13824,   679>,
        block_fft_implementation<12,   12,   12,  144,     2, 27648,   680>
    >;
};

template<> struct block_fft_record<13, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,    1,    32,     0,   681>
    >;
};

template<> struct block_fft_record<169, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,   13,    22,  1352,   682>,
        block_fft_implementation<13,   13,   13,   13,    22,  2704,   683>
    >;
};

template<> struct block_fft_record<2197, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,  169,     3, 17576,   684>,
        block_fft_implementation<13,   13,   13,  169,     1, 35152,   685>
    >;
};

template<> struct block_fft_record<14, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,    1,    32,     0,   686>
    >;
};

template<> struct block_fft_record<196, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,   14,    16,  1568,   687>,
        block_fft_implementation<14,   14,   14,   14,    16,  3136,   688>
    >;
};

template<> struct block_fft_record<2744, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,  196,     2, 21952,   690>,
        block_fft_implementation<14,   14,   14,  196,     1, 43904,   689>
    >;
};

template<> struct block_fft_record<15, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,    1,    32,     0,   691>
    >;
};

template<> struct block_fft_record<225, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,   15,    17,  1800,   692>,
        block_fft_implementation<15,   15,   15,   15,    17,  3600,   693>
    >;
};

template<> struct block_fft_record<3375, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,  225,     2, 27000,   694>,
        block_fft_implementation<15,   15,   15,  225,     1, 54000,   695>
    >;
};

template<> struct block_fft_record<17, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,    1,    32,     0,   696>
    >;
};

template<> struct block_fft_record<289, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,   17,    13,  2312,   697>,
        block_fft_implementation<17,   17,   17,   17,    13,  4624,   698>
    >;
};

template<> struct block_fft_record<4913, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,  289,     1, 39304,   699>
    >;
};

template<> struct block_fft_record<18, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,    1,    32,     0,   701>
    >;
};

template<> struct block_fft_record<324, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,   18,     7,  2592,   702>,
        block_fft_implementation<18,   18,   18,   18,     7,  5184,   703>
    >;
};

template<> struct block_fft_record<5832, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,  324,     1, 46656,   704>
    >;
};

template<> struct block_fft_record<19, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,    1,    32,     0,   706>
    >;
};

template<> struct block_fft_record<361, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,   19,     5,  2888,   707>,
        block_fft_implementation<19,   19,   19,   19,     5,  5776,   708>
    >;
};

template<> struct block_fft_record<6859, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,  361,     1, 54872,   709>
    >;
};

template<> struct block_fft_record<20, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,    1,    32,     0,   710>
    >;
};

template<> struct block_fft_record<400, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,   20,     8,  3200,   711>,
        block_fft_implementation<20,   20,   20,   20,     8,  6400,   712>
    >;
};

template<> struct block_fft_record<8000, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,  400,     1, 64000,   713>
    >;
};

template<> struct block_fft_record<21, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,   21,    1,    32,     0,   714>
    >;
};

template<> struct block_fft_record<441, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,   21,   21,     3,  3528,   715>,
        block_fft_implementation<21,   21,   21,   21,     3,  7056,   716>
    >;
};

template<> struct block_fft_record<22, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,   22,    1,    32,     0,   718>
    >;
};

template<> struct block_fft_record<484, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,   22,   22,    13,  3872,   719>,
        block_fft_implementation<22,   22,   22,   22,     7,  7744,   720>
    >;
};

template<> struct block_fft_record<23, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,   23,    1,    32,     0,   722>
    >;
};

template<> struct block_fft_record<529, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,   23,   23,    11,  4232,   724>,
        block_fft_implementation<23,   23,   23,   23,     4,  8464,   723>
    >;
};

template<> struct block_fft_record<24, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,   24,    1,    32,     0,   726>
    >;
};

template<> struct block_fft_record<576, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,   24,   24,     4,  4608,   727>,
        block_fft_implementation<24,   24,   24,   24,     4,  9216,   728>
    >;
};

template<> struct block_fft_record<26, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,   26,    1,    32,     0,   729>
    >;
};

template<> struct block_fft_record<676, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,   26,   26,    11,  5408,   730>,
        block_fft_implementation<26,   26,   26,   26,     6, 10816,   731>
    >;
};

template<> struct block_fft_record<28, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,   28,    1,    32,     0,   732>
    >;
};

template<> struct block_fft_record<784, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,   28,   28,     8,  6272,   733>,
        block_fft_implementation<28,   28,   28,   28,     2, 12544,   734>
    >;
};

template<> struct block_fft_record<29, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,   29,    1,    32,     0,   735>
    >;
};

template<> struct block_fft_record<841, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,   29,   29,     1,  6728,   736>,
        block_fft_implementation<29,   29,   29,   29,     1, 13456,   737>
    >;
};

template<> struct block_fft_record<30, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,   30,    1,    32,     0,   738>
    >;
};

template<> struct block_fft_record<900, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,   30,   30,     1,  7200,   739>,
        block_fft_implementation<30,   30,   30,   30,     1, 14400,   740>
    >;
};

template<> struct block_fft_record<31, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,   31,    1,    32,     0,   741>
    >;
};

template<> struct block_fft_record<961, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,   31,   31,     1,  7688,   742>,
        block_fft_implementation<31,   31,   31,   31,     1, 15376,   743>
    >;
};

template<> struct block_fft_record<44, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,    2,   22,    2,    16,   352,   744>,
        block_fft_implementation<22,    2,   22,    2,    32,   704,   745>
    >;
};

template<> struct block_fft_record<48, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,    4,   12,    4,     8,   384,   746>,
        block_fft_implementation<12,    4,   12,    4,     8,   768,   747>,
        block_fft_implementation< 6,    2,    6,    8,     4,   384,   751>,
        block_fft_implementation< 6,    2,    6,    8,     4,   768,   748>,
        block_fft_implementation<24,    2,   24,    2,    16,   384,   749>,
        block_fft_implementation<24,    2,   24,    2,    16,   768,   750>
    >;
};

template<> struct block_fft_record<52, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,    2,   26,    2,    32,   416,   752>,
        block_fft_implementation<26,    2,   26,    2,    16,   832,   753>
    >;
};

template<> struct block_fft_record<56, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,    2,   14,    4,     8,   448,   755>,
        block_fft_implementation<28,    2,   28,    2,    16,   448,   754>,
        block_fft_implementation<14,    2,   14,    4,    16,   896,   756>,
        block_fft_implementation<28,    2,   28,    2,    16,   896,   757>
    >;
};

template<> struct block_fft_record<60, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,    2,   30,    2,    16,   480,   758>,
        block_fft_implementation<30,    2,   30,    2,    16,   960,   759>
    >;
};

template<> struct block_fft_record<68, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    2,    16,   544,   760>,
        block_fft_implementation<34,    2,   34,    2,    16,  1088,   761>
    >;
};

template<> struct block_fft_record<72, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,    6,   12,    6,    16,   576,   762>,
        block_fft_implementation<24,    3,   24,    3,    32,   576,   764>,
        block_fft_implementation<12,    6,   12,    6,    16,  1152,   763>,
        block_fft_implementation<18,    2,   18,    4,     8,   576,   766>,
        block_fft_implementation<36,    2,   36,    2,    16,   576,   765>,
        block_fft_implementation< 6,    2,    6,   12,     8,   576,   767>,
        block_fft_implementation<18,    2,   18,    4,     8,  1152,   769>,
        block_fft_implementation<36,    2,   36,    2,    16,  1152,   771>,
        block_fft_implementation< 6,    2,    6,   12,     8,  1152,   768>,
        block_fft_implementation<24,    3,   24,    3,    10,  1152,   770>
    >;
};

template<> struct block_fft_record<76, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    2,    16,   608,   772>,
        block_fft_implementation<38,    2,   38,    2,    16,  1216,   773>
    >;
};

template<> struct block_fft_record<80, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,    4,   20,    4,     8,   640,   774>,
        block_fft_implementation<20,    4,   20,    4,     8,  1280,   775>,
        block_fft_implementation<10,    2,   10,    8,     4,   640,   778>,
        block_fft_implementation<10,    2,   10,    8,     4,  1280,   776>,
        block_fft_implementation<40,    2,   40,    2,    16,   640,   777>,
        block_fft_implementation<40,    2,   40,    2,    16,  1280,   779>
    >;
};

template<> struct block_fft_record<84, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    2,    16,   672,   780>,
        block_fft_implementation<42,    2,   42,    2,    16,  1344,   781>
    >;
};

template<> struct block_fft_record<88, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,    2,   22,    4,     8,   704,   782>,
        block_fft_implementation<44,    2,   44,    2,    32,   704,   783>,
        block_fft_implementation<22,    2,   22,    4,    16,  1408,   784>,
        block_fft_implementation<44,    2,   44,    2,    16,  1408,   785>
    >;
};

template<> struct block_fft_record<92, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    2,    32,   736,   786>,
        block_fft_implementation<46,    2,   46,    2,    16,  1472,   787>
    >;
};

template<> struct block_fft_record<96, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,    4,   24,    4,     8,   768,   788>,
        block_fft_implementation< 6,    2,    6,   16,     2,   768,   792>,
        block_fft_implementation<12,    2,   12,    8,     4,   768,   789>,
        block_fft_implementation< 6,    2,    6,   16,     2,  1536,   793>,
        block_fft_implementation<24,    4,   24,    4,     8,  1536,   791>,
        block_fft_implementation<12,    2,   12,    8,     4,  1536,   790>,
        block_fft_implementation<48,    2,   48,    2,    16,   768,   794>,
        block_fft_implementation<48,    2,   48,    2,    16,  1536,   795>
    >;
};

template<> struct block_fft_record<104, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,    2,   26,    4,     8,   832,   796>,
        block_fft_implementation<52,    2,   52,    2,    16,   832,   797>,
        block_fft_implementation<26,    2,   26,    4,     8,  1664,   798>,
        block_fft_implementation<52,    2,   52,    2,    16,  1664,   799>
    >;
};

template<> struct block_fft_record<108, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,    6,   18,    6,    16,   864,   800>,
        block_fft_implementation<12,    3,   12,    9,     7,   864,   806>,
        block_fft_implementation<36,    3,   36,    3,    32,   864,   803>,
        block_fft_implementation<18,    6,   18,    6,    16,  1728,   801>,
        block_fft_implementation< 6,    3,    6,   18,    16,   864,   802>,
        block_fft_implementation<12,    3,   12,    9,     7,  1728,   807>,
        block_fft_implementation<54,    2,   54,    2,    16,   864,   805>,
        block_fft_implementation< 6,    3,    6,   18,    16,  1728,   804>,
        block_fft_implementation<36,    3,   36,    3,    32,  1728,   808>,
        block_fft_implementation<54,    2,   54,    2,    16,  1728,   809>
    >;
};

template<> struct block_fft_record<112, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,    4,   28,    4,     8,   896,   810>,
        block_fft_implementation<28,    4,   28,    4,     8,  1792,   812>,
        block_fft_implementation<14,    2,   14,    8,     4,   896,   813>,
        block_fft_implementation<56,    2,   56,    2,    16,   896,   814>,
        block_fft_implementation<14,    2,   14,    8,     8,  1792,   811>,
        block_fft_implementation<56,    2,   56,    2,    16,  1792,   815>
    >;
};

template<> struct block_fft_record<116, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    2,    16,   928,   816>,
        block_fft_implementation<58,    2,   58,    2,    16,  1856,   817>
    >;
};

template<> struct block_fft_record<120, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,    2,   30,    4,     8,   960,   818>,
        block_fft_implementation<60,    2,   60,    2,    16,   960,   819>,
        block_fft_implementation<30,    2,   30,    4,     8,  1920,   820>,
        block_fft_implementation<60,    2,   60,    2,    16,  1920,   821>
    >;
};

template<> struct block_fft_record<124, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    2,    16,   992,   822>,
        block_fft_implementation<62,    2,   62,    2,    16,  1984,   823>
    >;
};

template<> struct block_fft_record<136, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    4,     8,  1088,   824>,
        block_fft_implementation<34,    2,   34,    4,     8,  2176,   825>
    >;
};

template<> struct block_fft_record<152, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    4,     8,  1216,   826>,
        block_fft_implementation<38,    2,   38,    4,     8,  2432,   827>
    >;
};

template<> struct block_fft_record<160, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<40,    4,   40,    4,     8,  1280,   829>,
        block_fft_implementation<10,    2,   10,   16,     2,  1280,   832>,
        block_fft_implementation<40,    4,   40,    4,     8,  2560,   833>,
        block_fft_implementation<20,    2,   20,    8,     4,  1280,   828>,
        block_fft_implementation<10,    2,   10,   16,     2,  2560,   831>,
        block_fft_implementation<20,    2,   20,    8,     4,  2560,   830>
    >;
};

template<> struct block_fft_record<168, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    4,     8,  1344,   834>,
        block_fft_implementation<42,    2,   42,    4,     8,  2688,   835>
    >;
};

template<> struct block_fft_record<176, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,    4,   44,    4,    16,  1408,   836>,
        block_fft_implementation<44,    4,   44,    4,     8,  2816,   839>,
        block_fft_implementation<22,    2,   22,    8,     4,  1408,   838>,
        block_fft_implementation<22,    2,   22,    8,     8,  2816,   837>
    >;
};

template<> struct block_fft_record<180, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,    6,   30,    6,    16,  1440,   840>,
        block_fft_implementation<60,    3,   60,    3,    21,  1440,   842>,
        block_fft_implementation<30,    6,   30,    6,    16,  2880,   841>,
        block_fft_implementation<60,    3,   60,    3,    21,  2880,   843>
    >;
};

template<> struct block_fft_record<184, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    4,    16,  1472,   844>,
        block_fft_implementation<46,    2,   46,    4,     8,  2944,   845>
    >;
};

template<> struct block_fft_record<192, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,    8,   24,    8,     4,  1536,   847>,
        block_fft_implementation<12,    4,   12,   16,     2,  1536,   846>,
        block_fft_implementation<12,    4,   12,   16,     2,  3072,   848>,
        block_fft_implementation< 6,    2,    6,   32,     1,  1536,   851>,
        block_fft_implementation<24,    8,   24,    8,     4,  3072,   849>,
        block_fft_implementation<48,    4,   48,    4,     8,  1536,   850>,
        block_fft_implementation<48,    4,   48,    4,     8,  3072,   853>,
        block_fft_implementation< 6,    2,    6,   32,     1,  3072,   852>
    >;
};

template<> struct block_fft_record<200, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   10,   20,   10,    16,  1600,   854>,
        block_fft_implementation<40,    5,   40,    5,    19,  1600,   857>,
        block_fft_implementation<20,   10,   20,   10,    16,  3200,   855>,
        block_fft_implementation<50,    2,   50,    4,    16,  1600,   859>,
        block_fft_implementation<50,    2,   50,    4,     8,  3200,   861>,
        block_fft_implementation<10,    2,   10,   20,     8,  1600,   856>,
        block_fft_implementation<40,    5,   40,    5,     6,  3200,   860>,
        block_fft_implementation<10,    2,   10,   20,     8,  3200,   858>
    >;
};

template<> struct block_fft_record<208, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,    4,   52,    4,     8,  1664,   862>,
        block_fft_implementation<52,    4,   52,    4,     8,  3328,   865>,
        block_fft_implementation<26,    2,   26,    8,     4,  1664,   864>,
        block_fft_implementation<26,    2,   26,    8,     4,  3328,   863>
    >;
};

template<> struct block_fft_record<224, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<56,    4,   56,    4,    16,  1792,   868>,
        block_fft_implementation<14,    2,   14,   16,     2,  1792,   867>,
        block_fft_implementation<28,    2,   28,    8,     4,  1792,   866>,
        block_fft_implementation<14,    2,   14,   16,     4,  3584,   869>,
        block_fft_implementation<28,    2,   28,    8,     4,  3584,   870>,
        block_fft_implementation<56,    4,   56,    4,     8,  3584,   871>
    >;
};

template<> struct block_fft_record<232, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    4,     8,  1856,   872>,
        block_fft_implementation<58,    2,   58,    4,     8,  3712,   873>
    >;
};

template<> struct block_fft_record<240, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,    4,   60,    4,     8,  1920,   874>,
        block_fft_implementation<30,    2,   30,    8,     4,  1920,   875>,
        block_fft_implementation<60,    4,   60,    4,     8,  3840,   877>,
        block_fft_implementation<30,    2,   30,    8,     4,  3840,   876>
    >;
};

template<> struct block_fft_record<248, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    4,     8,  1984,   878>,
        block_fft_implementation<62,    2,   62,    4,     8,  3968,   879>
    >;
};

template<> struct block_fft_record<252, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    6,   42,    6,    16,  2016,   880>,
        block_fft_implementation<42,    6,   42,    6,    16,  4032,   881>
    >;
};

template<> struct block_fft_record<272, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    8,     4,  2176,   882>,
        block_fft_implementation<34,    2,   34,    8,     4,  4352,   883>
    >;
};

template<> struct block_fft_record<288, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   12,   24,   12,     8,  2304,   884>,
        block_fft_implementation<36,    2,   36,    8,     4,  2304,   886>,
        block_fft_implementation<18,    2,   18,   16,     2,  2304,   888>,
        block_fft_implementation< 6,    2,    6,   48,     2,  2304,   889>,
        block_fft_implementation<18,    2,   18,   16,     2,  4608,   890>,
        block_fft_implementation<48,    6,   48,    6,    16,  2304,   892>,
        block_fft_implementation<36,    2,   36,    8,     4,  4608,   894>,
        block_fft_implementation<24,   12,   24,   12,     5,  4608,   891>,
        block_fft_implementation< 6,    2,    6,   48,     2,  4608,   893>,
        block_fft_implementation<12,    2,   12,   24,     4,  2304,   885>,
        block_fft_implementation<48,    6,   48,    6,     5,  4608,   895>,
        block_fft_implementation<12,    2,   12,   24,     4,  4608,   887>
    >;
};

template<> struct block_fft_record<300, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   10,   30,   10,    16,  2400,   896>,
        block_fft_implementation<60,    5,   60,    5,    19,  2400,   897>,
        block_fft_implementation<30,   10,   30,   10,     3,  4800,   898>,
        block_fft_implementation<60,    5,   60,    5,    12,  4800,   899>
    >;
};

template<> struct block_fft_record<304, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    8,     4,  2432,   900>,
        block_fft_implementation<38,    2,   38,    8,     4,  4864,   901>
    >;
};

template<> struct block_fft_record<320, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<40,    8,   40,    8,     4,  2560,   903>,
        block_fft_implementation<20,    4,   20,   16,     2,  2560,   902>,
        block_fft_implementation<40,    8,   40,    8,     4,  5120,   907>,
        block_fft_implementation<20,    4,   20,   16,     2,  5120,   904>,
        block_fft_implementation<10,    2,   10,   32,     1,  2560,   906>,
        block_fft_implementation<10,    2,   10,   32,     1,  5120,   905>
    >;
};

template<> struct block_fft_record<336, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    8,     4,  2688,   908>,
        block_fft_implementation<42,    2,   42,    8,     4,  5376,   909>
    >;
};

template<> struct block_fft_record<352, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,    2,   44,    8,     8,  2816,   910>,
        block_fft_implementation<22,    2,   22,   16,     2,  2816,   912>,
        block_fft_implementation<22,    2,   22,   16,     4,  5632,   911>,
        block_fft_implementation<44,    2,   44,    8,     4,  5632,   913>
    >;
};

template<> struct block_fft_record<360, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,    6,   60,    6,    16,  2880,   915>,
        block_fft_implementation<30,    2,   30,   12,     8,  2880,   914>,
        block_fft_implementation<30,    2,   30,   12,     8,  5760,   916>,
        block_fft_implementation<60,    6,   60,    6,    10,  5760,   917>
    >;
};

template<> struct block_fft_record<368, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    8,     8,  2944,   918>,
        block_fft_implementation<46,    2,   46,    8,     4,  5888,   919>
    >;
};

template<> struct block_fft_record<384, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<48,    8,   48,    8,     4,  3072,   923>,
        block_fft_implementation<48,    8,   48,    8,     4,  6144,   927>,
        block_fft_implementation< 6,    2,    6,   64,     1,  3072,   925>,
        block_fft_implementation<12,    2,   12,   32,     1,  3072,   920>,
        block_fft_implementation< 6,    2,    6,   64,     1,  6144,   926>,
        block_fft_implementation<12,    2,   12,   32,     1,  6144,   922>,
        block_fft_implementation<24,    2,   24,   16,     2,  3072,   921>,
        block_fft_implementation<24,    2,   24,   16,     2,  6144,   924>
    >;
};

template<> struct block_fft_record<392, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   14,   28,   14,     9,  3136,   928>,
        block_fft_implementation<56,    7,   56,    7,     9,  3136,   932>,
        block_fft_implementation<28,   14,   28,   14,     9,  6272,   931>,
        block_fft_implementation<14,    2,   14,   28,     8,  3136,   929>,
        block_fft_implementation<14,    2,   14,   28,     8,  6272,   930>,
        block_fft_implementation<56,    7,   56,    7,     8,  6272,   933>
    >;
};

template<> struct block_fft_record<416, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,    2,   26,   16,     2,  3328,   935>,
        block_fft_implementation<52,    2,   52,    8,     4,  3328,   934>,
        block_fft_implementation<26,    2,   26,   16,     2,  6656,   936>,
        block_fft_implementation<52,    2,   52,    8,     4,  6656,   937>
    >;
};

template<> struct block_fft_record<432, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<36,   12,   36,   12,     8,  3456,   938>,
        block_fft_implementation<36,   12,   36,   12,     8,  6912,   947>,
        block_fft_implementation<54,    2,   54,    8,     4,  3456,   948>,
        block_fft_implementation<12,    3,   12,   36,     8,  3456,   939>,
        block_fft_implementation<24,    3,   24,   18,     7,  3456,   941>,
        block_fft_implementation<18,    2,   18,   24,     4,  3456,   943>,
        block_fft_implementation< 6,    2,    6,   72,     4,  3456,   940>,
        block_fft_implementation<12,    3,   12,   36,     8,  6912,   942>,
        block_fft_implementation<48,    3,   48,    9,     7,  3456,   949>,
        block_fft_implementation<18,    2,   18,   24,     4,  6912,   944>,
        block_fft_implementation< 6,    2,    6,   72,     4,  6912,   945>,
        block_fft_implementation<48,    3,   48,    9,     7,  6912,   951>,
        block_fft_implementation<24,    3,   24,   18,     7,  6912,   946>,
        block_fft_implementation<54,    2,   54,    8,     4,  6912,   950>
    >;
};

template<> struct block_fft_record<448, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<56,    8,   56,    8,     4,  3584,   953>,
        block_fft_implementation<28,    4,   28,   16,     2,  3584,   952>,
        block_fft_implementation<56,    8,   56,    8,     4,  7168,   957>,
        block_fft_implementation<28,    4,   28,   16,     2,  7168,   954>,
        block_fft_implementation<14,    2,   14,   32,     1,  3584,   956>,
        block_fft_implementation<14,    2,   14,   32,     2,  7168,   955>
    >;
};

template<> struct block_fft_record<464, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    8,     4,  3712,   958>,
        block_fft_implementation<58,    2,   58,    8,     4,  7424,   959>
    >;
};

template<> struct block_fft_record<480, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,    2,   30,   16,     2,  3840,   960>,
        block_fft_implementation<60,    2,   60,    8,     4,  3840,   961>,
        block_fft_implementation<30,    2,   30,   16,     2,  7680,   962>,
        block_fft_implementation<60,    2,   60,    8,     4,  7680,   963>
    >;
};

template<> struct block_fft_record<496, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    8,     4,  3968,   964>,
        block_fft_implementation<62,    2,   62,    8,     4,  7936,   965>
    >;
};

template<> struct block_fft_record<500, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<50,   10,   50,   10,    16,  4000,   967>,
        block_fft_implementation<20,    5,   20,   25,     5,  4000,   968>,
        block_fft_implementation<10,    5,   10,   50,     7,  4000,   966>,
        block_fft_implementation<10,    5,   10,   50,     7,  8000,   969>,
        block_fft_implementation<50,   10,   50,   10,     3,  8000,   971>,
        block_fft_implementation<20,    5,   20,   25,     5,  8000,   970>
    >;
};

template<> struct block_fft_record<504, double, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,   12,     8,  4032,   972>,
        block_fft_implementation<42,    2,   42,   12,     8,  8064,   973>
    >;
};
