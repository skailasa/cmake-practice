//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<33, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<33,   33,   33,    1,    32,     0,     0>
    >;
};

template<> struct block_fft_record<34, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,   34,   34,    1,    32,     0,     1>
    >;
};

template<> struct block_fft_record<35, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<35,   35,   35,    1,    32,     0,     2>
    >;
};

template<> struct block_fft_record<37, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<37,   37,   37,    1,    32,     0,     3>
    >;
};

template<> struct block_fft_record<38, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,   38,   38,    1,    32,     0,     4>
    >;
};

template<> struct block_fft_record<39, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<39,   39,   39,    1,    32,     0,     5>
    >;
};

template<> struct block_fft_record<40, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,    2,   20,    2,   256,   160,     6>,
        block_fft_implementation<10,    2,   10,    4,    64,   320,     7>,
        block_fft_implementation<10,    2,   10,    4,    64,   160,     8>,
        block_fft_implementation<40,   40,   40,    1,    32,     0,     9>
    >;
};

template<> struct block_fft_record<41, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<41,   41,   41,    1,    32,     0,    10>
    >;
};

template<> struct block_fft_record<42, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,   42,   42,    1,    32,     0,    11>
    >;
};

template<> struct block_fft_record<43, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<43,   43,   43,    1,    32,     0,    12>
    >;
};

template<> struct block_fft_record<44, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,   44,   44,    1,    32,     0,    13>
    >;
};

template<> struct block_fft_record<45, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<45,   45,   45,    1,    32,     0,    14>
    >;
};

template<> struct block_fft_record<46, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,   46,   46,    1,    32,     0,    15>
    >;
};

template<> struct block_fft_record<47, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<47,   47,   47,    1,    32,     0,    16>
    >;
};

template<> struct block_fft_record<48, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<48,   48,   48,    1,    32,     0,    17>
    >;
};

template<> struct block_fft_record<50, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<50,   50,   50,    1,    32,     0,    18>
    >;
};

template<> struct block_fft_record<51, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<51,   51,   51,    1,    32,     0,    19>
    >;
};

template<> struct block_fft_record<52, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,   52,   52,    1,    32,     0,    20>
    >;
};

template<> struct block_fft_record<53, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<53,   53,   53,    1,    32,     0,    21>
    >;
};

template<> struct block_fft_record<54, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<54,   54,   54,    1,    32,     0,    22>
    >;
};

template<> struct block_fft_record<55, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<55,   55,   55,    1,    32,     0,    23>
    >;
};

template<> struct block_fft_record<56, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<56,   56,   56,    1,    32,     0,    24>
    >;
};

template<> struct block_fft_record<57, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<57,   57,   57,    1,    32,     0,    25>
    >;
};

template<> struct block_fft_record<58, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,   58,   58,    1,    32,     0,    26>
    >;
};

template<> struct block_fft_record<59, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<59,   59,   59,    1,    32,     0,    27>
    >;
};

template<> struct block_fft_record<60, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,   60,   60,    1,    32,     0,    28>
    >;
};

template<> struct block_fft_record<61, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<61,   61,   61,    1,    32,     0,    29>
    >;
};

template<> struct block_fft_record<62, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,   62,   62,    1,    32,     0,    30>
    >;
};

template<> struct block_fft_record<63, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<63,   63,   63,    1,    32,     0,    31>
    >;
};

template<> struct block_fft_record<2, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    2,    1,   512,     0,    32>
    >;
};

template<> struct block_fft_record<4, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    4,    1,   256,     0,    33>,
        block_fft_implementation< 2,    2,    2,    2,   240,    32,    35>,
        block_fft_implementation< 2,    2,    2,    2,   256,    16,    34>
    >;
};

template<> struct block_fft_record<8, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    1,   512,     0,    36>,
        block_fft_implementation< 4,    2,    4,    2,   128,    32,    37>,
        block_fft_implementation< 4,    2,    4,    2,   112,    64,    38>,
        block_fft_implementation< 2,    2,    2,    4,   128,    64,    40>,
        block_fft_implementation< 2,    2,    2,    4,   256,    32,    39>
    >;
};

template<> struct block_fft_record<16, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,    1,   256,     0,    41>,
        block_fft_implementation< 4,    4,    4,    4,    64,    64,    42>,
        block_fft_implementation< 4,    4,    4,    4,    64,   128,    43>,
        block_fft_implementation< 8,    2,    8,    2,   208,    64,    44>,
        block_fft_implementation< 8,    2,    8,    2,    30,   128,    45>,
        block_fft_implementation< 2,    2,    2,    8,    64,   128,    47>,
        block_fft_implementation< 2,    2,    2,    8,   112,    64,    46>
    >;
};

template<> struct block_fft_record<32, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<32,   32,   32,    1,   512,     0,    48>,
        block_fft_implementation< 8,    4,    8,    4,    16,   256,    49>,
        block_fft_implementation< 8,    4,    8,    4,   176,   128,    50>,
        block_fft_implementation< 4,    2,    4,    8,    24,   256,    51>,
        block_fft_implementation< 4,    2,    4,    8,    48,   128,    52>,
        block_fft_implementation< 2,    2,    2,   16,    28,   256,    53>,
        block_fft_implementation<16,    2,   16,    2,    14,   256,    54>,
        block_fft_implementation< 2,    2,    2,   16,    64,   128,    55>,
        block_fft_implementation<16,    2,   16,    2,    27,   128,    56>
    >;
};

template<> struct block_fft_record<64, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    8,    16,   256,    58>,
        block_fft_implementation< 8,    8,    8,    8,     8,   512,    57>,
        block_fft_implementation< 4,    4,    4,   16,     8,   512,    59>,
        block_fft_implementation< 4,    4,    4,   16,    29,   256,    60>,
        block_fft_implementation<16,    4,   16,    4,    23,   256,    62>,
        block_fft_implementation<16,    4,   16,    4,     8,   512,    61>,
        block_fft_implementation<32,    2,   32,    2,    14,   512,    63>,
        block_fft_implementation< 2,    2,    2,   32,    14,   512,    64>,
        block_fft_implementation< 2,    2,    2,   32,    30,   256,    65>,
        block_fft_implementation<32,    2,   32,    2,    13,   256,    66>,
        block_fft_implementation<64,   64,   64,    1,   256,     0,    67>
    >;
};

template<> struct block_fft_record<128, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    8,   16,    8,     8,   512,    68>,
        block_fft_implementation< 8,    2,    8,   16,     6,  1024,    69>,
        block_fft_implementation<16,    8,   16,    8,     4,  1024,    70>,
        block_fft_implementation< 8,    2,    8,   16,     8,   512,    71>,
        block_fft_implementation< 4,    2,    4,   32,     8,  1024,    72>,
        block_fft_implementation< 4,    2,    4,   32,     8,   512,    73>,
        block_fft_implementation<32,    4,   32,    4,     8,  1024,    74>,
        block_fft_implementation<32,    4,   32,    4,     7,   512,    75>,
        block_fft_implementation< 2,    2,    2,   64,     8,  1024,    76>,
        block_fft_implementation< 2,    2,    2,   64,    15,   512,    77>
    >;
};

template<> struct block_fft_record<256, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,   16,     4,  1024,    78>,
        block_fft_implementation< 8,    4,    8,   32,     4,  1024,    80>,
        block_fft_implementation< 8,    4,    8,   32,     3,  2048,    79>,
        block_fft_implementation<16,   16,   16,   16,     2,  2048,    81>,
        block_fft_implementation<32,    8,   32,    8,     4,  2048,    82>,
        block_fft_implementation< 4,    4,    4,   64,     4,  2048,    83>,
        block_fft_implementation< 4,    4,    4,   64,     4,  1024,    84>,
        block_fft_implementation<32,    8,   32,    8,     4,  1024,    85>,
        block_fft_implementation< 2,    2,    2,  128,     8,  2048,    86>,
        block_fft_implementation< 2,    2,    2,  128,     4,  1024,    87>
    >;
};

template<> struct block_fft_record<512, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,   64,     2,  2048,    89>,
        block_fft_implementation<16,    2,   16,   32,     2,  2048,    90>,
        block_fft_implementation< 8,    8,    8,   64,     1,  4096,    88>,
        block_fft_implementation<32,   16,   32,   16,     2,  4096,    91>,
        block_fft_implementation<32,   16,   32,   16,     2,  2048,    92>,
        block_fft_implementation<16,    2,   16,   32,     1,  4096,    93>,
        block_fft_implementation< 4,    2,    4,  128,     2,  4096,    94>,
        block_fft_implementation< 4,    2,    4,  128,     2,  2048,    95>,
        block_fft_implementation< 2,    2,    2,  256,     4,  4096,    96>,
        block_fft_implementation< 2,    2,    2,  256,     2,  2048,    97>
    >;
};

template<> struct block_fft_record<1024, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    4,   16,   64,     1,  4096,    98>,
        block_fft_implementation<32,   32,   32,   32,     2,  4096,   102>,
        block_fft_implementation< 8,    2,    8,  128,     1,  8192,    99>,
        block_fft_implementation<32,   32,   32,   32,     1,  8192,   100>,
        block_fft_implementation< 8,    2,    8,  128,     1,  4096,   101>,
        block_fft_implementation<16,    4,   16,   64,     1,  8192,   103>,
        block_fft_implementation< 4,    4,    4,  256,     1,  8192,   104>,
        block_fft_implementation< 4,    4,    4,  256,     1,  4096,   105>,
        block_fft_implementation< 2,    2,    2,  512,     2,  8192,   106>,
        block_fft_implementation< 2,    2,    2,  512,     1,  4096,   107>
    >;
};

template<> struct block_fft_record<2048, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    8,   16,  128,     1,  8192,   108>,
        block_fft_implementation< 8,    4,    8,  256,     1, 16384,   109>,
        block_fft_implementation< 8,    4,    8,  256,     1,  8192,   110>,
        block_fft_implementation<16,    8,   16,  128,     1, 16384,   111>,
        block_fft_implementation<32,    2,   32,   64,     1,  8192,   113>,
        block_fft_implementation<32,    2,   32,   64,     1, 16384,   112>,
        block_fft_implementation< 4,    2,    4,  512,     1, 16384,   114>,
        block_fft_implementation< 4,    2,    4,  512,     1,  8192,   115>,
        block_fft_implementation< 2,    2,    2, 1024,     1, 16384,   116>,
        block_fft_implementation< 2,    2,    2, 1024,     1,  8192,   117>
    >;
};

template<> struct block_fft_record<4096, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,  256,     1, 16384,   118>,
        block_fft_implementation< 8,    8,    8,  512,     1, 16384,   120>,
        block_fft_implementation< 8,    8,    8,  512,     1, 32768,   119>,
        block_fft_implementation<16,   16,   16,  256,     1, 32768,   121>,
        block_fft_implementation<32,    4,   32,  128,     1, 16384,   123>,
        block_fft_implementation<32,    4,   32,  128,     1, 32768,   122>,
        block_fft_implementation< 4,    4,    4, 1024,     1, 32768,   124>,
        block_fft_implementation< 4,    4,    4, 1024,     1, 16384,   125>
    >;
};

template<> struct block_fft_record<8192, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    2,   16,  512,     1, 32768,   126>,
        block_fft_implementation<32,    8,   32,  256,     1, 65536,   127>,
        block_fft_implementation<32,    8,   32,  256,     1, 32768,   128>,
        block_fft_implementation< 8,    2,    8, 1024,     1, 65536,   129>,
        block_fft_implementation< 8,    2,    8, 1024,     1, 32768,   131>,
        block_fft_implementation<16,    2,   16,  512,     1, 65536,   130>
    >;
};

template<> struct block_fft_record<16384, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    4,   16, 1024,     1, 65536,   132>,
        block_fft_implementation<32,   16,   32,  512,     1, 65536,   133>
    >;
};

template<> struct block_fft_record<3, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    3,    1,    64,     0,   134>
    >;
};

template<> struct block_fft_record<9, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    1,   512,     0,   135>,
        block_fft_implementation< 3,    3,    3,    3,   256,    72,   136>,
        block_fft_implementation< 3,    3,    3,    3,   128,    36,   137>
    >;
};

template<> struct block_fft_record<27, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,    1,   512,     0,   138>,
        block_fft_implementation< 9,    3,    9,    3,    21,   216,   140>,
        block_fft_implementation< 9,    3,    9,    3,    21,   108,   139>,
        block_fft_implementation< 3,    3,    3,    9,   112,   216,   141>,
        block_fft_implementation< 3,    3,    3,    9,    32,   108,   142>
    >;
};

template<> struct block_fft_record<81, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    9,     7,   324,   144>,
        block_fft_implementation< 9,    9,    9,    9,     7,   648,   143>,
        block_fft_implementation<27,    3,   27,    3,    21,   324,   145>,
        block_fft_implementation<27,    3,   27,    3,    21,   648,   146>,
        block_fft_implementation< 3,    3,    3,   27,    32,   648,   147>,
        block_fft_implementation< 3,    3,    3,   27,    14,   324,   148>
    >;
};

template<> struct block_fft_record<243, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,    9,   27,    9,     7,   972,   150>,
        block_fft_implementation<27,    9,   27,    9,     7,  1944,   149>,
        block_fft_implementation< 9,    3,    9,   27,    23,  1944,   151>,
        block_fft_implementation< 9,    3,    9,   27,     8,   972,   152>,
        block_fft_implementation< 3,    3,    3,   81,     6,  1944,   153>,
        block_fft_implementation< 3,    3,    3,   81,     3,   972,   154>
    >;
};

template<> struct block_fft_record<729, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,   27,     9,  5832,   155>,
        block_fft_implementation<27,   27,   27,   27,     9,  2916,   156>,
        block_fft_implementation< 9,    9,    9,   81,     3,  2916,   158>,
        block_fft_implementation< 9,    9,    9,   81,     3,  5832,   157>,
        block_fft_implementation< 3,    3,    3,  243,     2,  5832,   159>,
        block_fft_implementation< 3,    3,    3,  243,     1,  2916,   160>
    >;
};

template<> struct block_fft_record<2187, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,    3,   27,   81,     3, 17496,   162>,
        block_fft_implementation< 9,    3,    9,  243,     1, 17496,   161>,
        block_fft_implementation< 9,    3,    9,  243,     1,  8748,   163>,
        block_fft_implementation<27,    3,   27,   81,     3,  8748,   164>,
        block_fft_implementation< 3,    3,    3,  729,     1, 17496,   165>,
        block_fft_implementation< 3,    3,    3,  729,     1,  8748,   166>
    >;
};

template<> struct block_fft_record<6561, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,    9,   27,  243,     1, 52488,   168>,
        block_fft_implementation<27,    9,   27,  243,     1, 26244,   167>,
        block_fft_implementation< 9,    9,    9,  729,     1, 52488,   170>,
        block_fft_implementation< 9,    9,    9,  729,     1, 26244,   169>
    >;
};

template<> struct block_fft_record<5, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    5,    1,   256,     0,   172>
    >;
};

template<> struct block_fft_record<25, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,    1,   256,     0,   173>,
        block_fft_implementation< 5,    5,    5,    5,    96,   200,   174>,
        block_fft_implementation< 5,    5,    5,    5,    25,   100,   175>
    >;
};

template<> struct block_fft_record<125, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,    5,   25,    5,    25,   500,   176>,
        block_fft_implementation<25,    5,   25,    5,    24,  1000,   177>,
        block_fft_implementation< 5,    5,    5,   25,     5,  1000,   178>,
        block_fft_implementation< 5,    5,    5,   25,    10,   500,   179>
    >;
};

template<> struct block_fft_record<625, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,   25,     5,  2500,   180>,
        block_fft_implementation<25,   25,   25,   25,    10,  5000,   181>,
        block_fft_implementation< 5,    5,    5,  125,     1,  5000,   182>,
        block_fft_implementation< 5,    5,    5,  125,     1,  2500,   183>
    >;
};

template<> struct block_fft_record<3125, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,    5,   25,  125,     1, 12500,   184>,
        block_fft_implementation<25,    5,   25,  125,     1, 25000,   185>,
        block_fft_implementation< 5,    5,    5,  625,     1, 25000,   186>,
        block_fft_implementation< 5,    5,    5,  625,     1, 12500,   187>
    >;
};

template<> struct block_fft_record<15625, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,  625,     1, 62500,   188>
    >;
};

template<> struct block_fft_record<7, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,    1,   448,     0,   189>
    >;
};

template<> struct block_fft_record<49, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<49,   49,   49,    1,   256,     0,   190>,
        block_fft_implementation< 7,    7,    7,    7,    18,   392,   191>,
        block_fft_implementation< 7,    7,    7,    7,    13,   196,   192>
    >;
};

template<> struct block_fft_record<343, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,   49,     5,  2744,   193>,
        block_fft_implementation< 7,    7,    7,   49,     5,  1372,   194>
    >;
};

template<> struct block_fft_record<2401, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,  343,     1, 19208,   195>,
        block_fft_implementation< 7,    7,    7,  343,     1,  9604,   196>
    >;
};

template<> struct block_fft_record<11, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    1,   512,     0,   198>
    >;
};

template<> struct block_fft_record<121, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,   11,    11,   968,   199>,
        block_fft_implementation<11,   11,   11,   11,    11,   484,   200>
    >;
};

template<> struct block_fft_record<1331, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,  121,     1, 10648,   202>,
        block_fft_implementation<11,   11,   11,  121,     1,  5324,   201>
    >;
};

template<> struct block_fft_record<6, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    1,   512,     0,   203>
    >;
};

template<> struct block_fft_record<36, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    6,    21,   288,   205>,
        block_fft_implementation< 6,    6,    6,    6,    21,   144,   204>,
        block_fft_implementation<36,   36,   36,    1,   512,     0,   206>
    >;
};

template<> struct block_fft_record<216, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,   36,     6,  1728,   207>,
        block_fft_implementation< 6,    6,    6,   36,    14,   864,   208>
    >;
};

template<> struct block_fft_record<1296, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,  216,     1, 10368,   209>,
        block_fft_implementation< 6,    6,    6,  216,     1,  5184,   210>
    >;
};

template<> struct block_fft_record<7776, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<36,    6,   36,  216,     1, 62208,   211>,
        block_fft_implementation<36,    6,   36,  216,     1, 31104,   212>
    >;
};

template<> struct block_fft_record<10, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,    1,   448,     0,   213>
    >;
};

template<> struct block_fft_record<100, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,   10,     3,   800,   215>,
        block_fft_implementation<10,   10,   10,   10,    25,   400,   214>
    >;
};

template<> struct block_fft_record<1000, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,  100,     1,  8000,   216>,
        block_fft_implementation<10,   10,   10,  100,     1,  4000,   217>
    >;
};

template<> struct block_fft_record<10000, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10, 1000,     1, 40000,   219>
    >;
};

template<> struct block_fft_record<12, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,    1,   448,     0,   220>
    >;
};

template<> struct block_fft_record<144, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,   12,     5,   576,   222>,
        block_fft_implementation<12,   12,   12,   12,     5,  1152,   221>
    >;
};

template<> struct block_fft_record<1728, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,  144,     2,  6912,   224>,
        block_fft_implementation<12,   12,   12,  144,     3, 13824,   223>
    >;
};

template<> struct block_fft_record<13, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,    1,   512,     0,   225>
    >;
};

template<> struct block_fft_record<169, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,   13,    19,  1352,   226>,
        block_fft_implementation<13,   13,   13,   13,     9,   676,   227>
    >;
};

template<> struct block_fft_record<2197, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,  169,     3, 17576,   229>,
        block_fft_implementation<13,   13,   13,  169,     3,  8788,   228>
    >;
};

template<> struct block_fft_record<14, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,    1,   512,     0,   230>
    >;
};

template<> struct block_fft_record<196, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,   14,     9,   784,   232>,
        block_fft_implementation<14,   14,   14,   14,    18,  1568,   231>
    >;
};

template<> struct block_fft_record<2744, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,  196,     1, 10976,   234>,
        block_fft_implementation<14,   14,   14,  196,     1, 21952,   233>
    >;
};

template<> struct block_fft_record<15, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,    1,   448,     0,   235>
    >;
};

template<> struct block_fft_record<225, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,   15,     2,  1800,   236>,
        block_fft_implementation<15,   15,   15,   15,     2,   900,   237>
    >;
};

template<> struct block_fft_record<3375, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,  225,     1, 27000,   238>,
        block_fft_implementation<15,   15,   15,  225,     1, 13500,   239>
    >;
};

template<> struct block_fft_record<17, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,    1,   512,     0,   240>
    >;
};

template<> struct block_fft_record<289, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,   17,    15,  2312,   241>,
        block_fft_implementation<17,   17,   17,   17,     7,  1156,   242>
    >;
};

template<> struct block_fft_record<4913, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,  289,     1, 39304,   243>,
        block_fft_implementation<17,   17,   17,  289,     2, 19652,   244>
    >;
};

template<> struct block_fft_record<18, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,    1,   512,     0,   245>
    >;
};

template<> struct block_fft_record<324, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,   18,     7,  1296,   247>,
        block_fft_implementation<18,   18,   18,   18,     7,  2592,   246>
    >;
};

template<> struct block_fft_record<5832, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,  324,     1, 46656,   248>,
        block_fft_implementation<18,   18,   18,  324,     1, 23328,   249>
    >;
};

template<> struct block_fft_record<19, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,    1,   512,     0,   250>
    >;
};

template<> struct block_fft_record<361, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,   19,    20,  2888,   251>,
        block_fft_implementation<19,   19,   19,   19,     3,  1444,   252>
    >;
};

template<> struct block_fft_record<6859, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,  361,     1, 54872,   253>,
        block_fft_implementation<19,   19,   19,  361,     1, 27436,   254>
    >;
};

template<> struct block_fft_record<20, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,    1,   512,     0,   255>
    >;
};

template<> struct block_fft_record<400, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,   20,     6,  1600,   257>,
        block_fft_implementation<20,   20,   20,   20,     3,  3200,   256>
    >;
};

template<> struct block_fft_record<8000, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,  400,     1, 64000,   259>,
        block_fft_implementation<20,   20,   20,  400,     1, 32000,   258>
    >;
};

template<> struct block_fft_record<21, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,   21,    1,   512,     0,   260>
    >;
};

template<> struct block_fft_record<441, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,   21,   21,     3,  3528,   261>,
        block_fft_implementation<21,   21,   21,   21,     3,  1764,   262>
    >;
};

template<> struct block_fft_record<9261, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,   21,  441,     1, 37044,   263>
    >;
};

template<> struct block_fft_record<22, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,   22,    1,   256,     0,   265>
    >;
};

template<> struct block_fft_record<484, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,   22,   22,    11,  1936,   267>,
        block_fft_implementation<22,   22,   22,   22,    11,  3872,   266>
    >;
};

template<> struct block_fft_record<10648, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,   22,  484,     1, 42592,   269>
    >;
};

template<> struct block_fft_record<23, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,   23,    1,   320,     0,   270>
    >;
};

template<> struct block_fft_record<529, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,   23,   23,    11,  4232,   271>,
        block_fft_implementation<23,   23,   23,   23,    11,  2116,   272>
    >;
};

template<> struct block_fft_record<12167, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,   23,  529,     1, 48668,   274>
    >;
};

template<> struct block_fft_record<24, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,   24,    1,   256,     0,   275>
    >;
};

template<> struct block_fft_record<576, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,   24,   24,     5,  2304,   276>,
        block_fft_implementation<24,   24,   24,   24,     1,  4608,   277>
    >;
};

template<> struct block_fft_record<13824, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,   24,  576,     1, 55296,   278>
    >;
};

template<> struct block_fft_record<26, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,   26,    1,   256,     0,   279>
    >;
};

template<> struct block_fft_record<676, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,   26,   26,     9,  2704,   281>,
        block_fft_implementation<26,   26,   26,   26,     9,  5408,   280>
    >;
};

template<> struct block_fft_record<28, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,   28,    1,   256,     0,   283>
    >;
};

template<> struct block_fft_record<784, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,   28,   28,     1,  3136,   284>,
        block_fft_implementation<28,   28,   28,   28,     9,  6272,   285>
    >;
};

template<> struct block_fft_record<29, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,   29,    1,   192,     0,   287>
    >;
};

template<> struct block_fft_record<841, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,   29,   29,     1,  6728,   288>,
        block_fft_implementation<29,   29,   29,   29,     1,  3364,   289>
    >;
};

template<> struct block_fft_record<30, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,   30,    1,   512,     0,   291>
    >;
};

template<> struct block_fft_record<900, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,   30,   30,     1,  3600,   293>,
        block_fft_implementation<30,   30,   30,   30,     1,  7200,   292>
    >;
};

template<> struct block_fft_record<31, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,   31,    1,   384,     0,   294>
    >;
};

template<> struct block_fft_record<961, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,   31,   31,     8,  7688,   295>,
        block_fft_implementation<31,   31,   31,   31,     4,  3844,   296>
    >;
};

template<> struct block_fft_record<68, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    2,    16,   272,   297>,
        block_fft_implementation<34,    2,   34,    2,    15,   544,   298>
    >;
};

template<> struct block_fft_record<72, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,    6,   12,    6,    21,   288,   299>,
        block_fft_implementation<12,    6,   12,    6,     5,   576,   301>,
        block_fft_implementation<36,    2,   36,    2,   192,   288,   302>,
        block_fft_implementation<24,    3,   24,    3,    21,   288,   300>,
        block_fft_implementation<36,    2,   36,    2,   112,   576,   306>,
        block_fft_implementation<24,    3,   24,    3,    20,   576,   304>,
        block_fft_implementation<18,    2,   18,    4,    32,   576,   305>,
        block_fft_implementation<18,    2,   18,    4,    24,   288,   303>,
        block_fft_implementation< 6,    2,    6,   12,    28,   576,   308>,
        block_fft_implementation< 6,    2,    6,   12,    64,   288,   307>
    >;
};

template<> struct block_fft_record<76, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    2,    32,   304,   309>,
        block_fft_implementation<38,    2,   38,    2,    15,   608,   310>
    >;
};

template<> struct block_fft_record<80, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,    4,   20,    4,    16,   320,   311>,
        block_fft_implementation<20,    4,   20,    4,    96,   640,   312>,
        block_fft_implementation<40,    2,   40,    2,    96,   320,   313>,
        block_fft_implementation<40,    2,   40,    2,    96,   640,   314>,
        block_fft_implementation<10,    2,   10,    8,    64,   640,   316>,
        block_fft_implementation<10,    2,   10,    8,    32,   320,   315>
    >;
};

template<> struct block_fft_record<84, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    2,   192,   336,   317>,
        block_fft_implementation<42,    2,   42,    2,    16,   672,   318>
    >;
};

template<> struct block_fft_record<88, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,    2,   44,    2,   128,   352,   319>,
        block_fft_implementation<44,    2,   44,    2,    30,   704,   322>,
        block_fft_implementation<22,    2,   22,    4,    96,   352,   320>,
        block_fft_implementation<22,    2,   22,    4,     8,   704,   321>
    >;
};

template<> struct block_fft_record<92, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    2,   128,   368,   323>,
        block_fft_implementation<46,    2,   46,    2,    64,   736,   324>
    >;
};

template<> struct block_fft_record<96, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,    4,   24,    4,     7,   768,   325>,
        block_fft_implementation<12,    2,   12,    8,     8,   768,   326>,
        block_fft_implementation<24,    4,   24,    4,   144,   384,   327>,
        block_fft_implementation<12,    2,   12,    8,    32,   384,   329>,
        block_fft_implementation< 6,    2,    6,   16,    10,   768,   328>,
        block_fft_implementation<48,    2,   48,    2,    12,   768,   330>,
        block_fft_implementation< 6,    2,    6,   16,    20,   384,   331>,
        block_fft_implementation<48,    2,   48,    2,    96,   384,   332>
    >;
};

template<> struct block_fft_record<104, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,    2,   52,    2,   128,   416,   333>,
        block_fft_implementation<52,    2,   52,    2,    31,   832,   336>,
        block_fft_implementation<26,    2,   26,    4,     8,   416,   334>,
        block_fft_implementation<26,    2,   26,    4,    16,   832,   335>
    >;
};

template<> struct block_fft_record<108, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,    6,   18,    6,    21,   432,   337>,
        block_fft_implementation<18,    6,   18,    6,     5,   864,   338>,
        block_fft_implementation<36,    3,   36,    3,    21,   432,   339>,
        block_fft_implementation<54,    2,   54,    2,   112,   432,   340>,
        block_fft_implementation<54,    2,   54,    2,    16,   864,   346>,
        block_fft_implementation<36,    3,   36,    3,     9,   864,   342>,
        block_fft_implementation< 6,    3,    6,   18,    16,   864,   343>,
        block_fft_implementation< 6,    3,    6,   18,    21,   432,   341>,
        block_fft_implementation<12,    3,   12,    9,    24,   432,   344>,
        block_fft_implementation<12,    3,   12,    9,     7,   864,   345>
    >;
};

template<> struct block_fft_record<112, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,    4,   28,    4,     8,   448,   347>,
        block_fft_implementation<28,    4,   28,    4,    15,   896,   348>,
        block_fft_implementation<56,    2,   56,    2,    96,   448,   349>,
        block_fft_implementation<56,    2,   56,    2,    16,   896,   352>,
        block_fft_implementation<14,    2,   14,    8,    31,   896,   351>,
        block_fft_implementation<14,    2,   14,    8,    16,   448,   350>
    >;
};

template<> struct block_fft_record<116, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    2,    32,   464,   353>,
        block_fft_implementation<58,    2,   58,    2,    16,   928,   354>
    >;
};

template<> struct block_fft_record<120, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,    2,   60,    2,   128,   480,   355>,
        block_fft_implementation<30,    2,   30,    4,     8,   960,   357>,
        block_fft_implementation<30,    2,   30,    4,    16,   480,   356>,
        block_fft_implementation<60,    2,   60,    2,    16,   960,   358>
    >;
};

template<> struct block_fft_record<124, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    2,   112,   496,   359>,
        block_fft_implementation<62,    2,   62,    2,    16,   992,   360>
    >;
};

template<> struct block_fft_record<136, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    4,     8,   544,   361>,
        block_fft_implementation<34,    2,   34,    4,     8,  1088,   362>
    >;
};

template<> struct block_fft_record<152, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    4,     8,   608,   363>,
        block_fft_implementation<38,    2,   38,    4,     8,  1216,   364>
    >;
};

template<> struct block_fft_record<160, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,    2,   20,    8,     8,  1280,   365>,
        block_fft_implementation<40,    4,   40,    4,    16,   640,   367>,
        block_fft_implementation<40,    4,   40,    4,     7,  1280,   366>,
        block_fft_implementation<10,    2,   10,   16,    16,  1280,   368>,
        block_fft_implementation<20,    2,   20,    8,    12,   640,   369>,
        block_fft_implementation<10,    2,   10,   16,    16,   640,   370>
    >;
};

template<> struct block_fft_record<168, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    4,     8,   672,   371>,
        block_fft_implementation<42,    2,   42,    4,     8,  1344,   372>
    >;
};

template<> struct block_fft_record<176, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,    4,   44,    4,     8,   704,   373>,
        block_fft_implementation<44,    4,   44,    4,    15,  1408,   374>,
        block_fft_implementation<22,    2,   22,    8,    32,   704,   375>,
        block_fft_implementation<22,    2,   22,    8,     4,  1408,   376>
    >;
};

template<> struct block_fft_record<180, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,    6,   30,    6,    21,   720,   377>,
        block_fft_implementation<30,    6,   30,    6,     5,  1440,   378>,
        block_fft_implementation<60,    3,   60,    3,    10,   720,   379>,
        block_fft_implementation<60,    3,   60,    3,    10,  1440,   380>
    >;
};

template<> struct block_fft_record<184, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    4,     8,   736,   382>,
        block_fft_implementation<46,    2,   46,    4,    16,  1472,   381>
    >;
};

template<> struct block_fft_record<192, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,    8,   24,    8,     8,   768,   383>,
        block_fft_implementation<24,    8,   24,    8,     4,  1536,   384>,
        block_fft_implementation<12,    4,   12,   16,     2,  1536,   385>,
        block_fft_implementation<12,    4,   12,   16,     8,   768,   386>,
        block_fft_implementation<48,    4,   48,    4,     7,   768,   387>,
        block_fft_implementation<48,    4,   48,    4,     7,  1536,   388>,
        block_fft_implementation< 6,    2,    6,   32,     7,  1536,   389>,
        block_fft_implementation< 6,    2,    6,   32,     8,   768,   390>
    >;
};

template<> struct block_fft_record<200, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   10,   20,   10,    12,   800,   391>,
        block_fft_implementation<20,   10,   20,   10,     6,  1600,   392>,
        block_fft_implementation<40,    5,   40,    5,    12,   800,   393>,
        block_fft_implementation<10,    2,   10,   20,     3,  1600,   394>,
        block_fft_implementation<50,    2,   50,    4,     8,   800,   397>,
        block_fft_implementation<40,    5,   40,    5,     5,  1600,   395>,
        block_fft_implementation<10,    2,   10,   20,    11,   800,   396>,
        block_fft_implementation<50,    2,   50,    4,     8,  1600,   398>
    >;
};

template<> struct block_fft_record<208, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,    4,   52,    4,     8,   832,   399>,
        block_fft_implementation<52,    4,   52,    4,     8,  1664,   400>,
        block_fft_implementation<26,    2,   26,    8,     8,   832,   401>,
        block_fft_implementation<26,    2,   26,    8,     8,  1664,   402>
    >;
};

template<> struct block_fft_record<224, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<56,    4,   56,    4,    64,   896,   404>,
        block_fft_implementation<28,    2,   28,    8,     8,  1792,   403>,
        block_fft_implementation<56,    4,   56,    4,    16,  1792,   407>,
        block_fft_implementation<14,    2,   14,   16,     2,  1792,   405>,
        block_fft_implementation<28,    2,   28,    8,    48,   896,   406>,
        block_fft_implementation<14,    2,   14,   16,    12,   896,   408>
    >;
};

template<> struct block_fft_record<232, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    4,     8,   928,   409>,
        block_fft_implementation<58,    2,   58,    4,     8,  1856,   410>
    >;
};

template<> struct block_fft_record<240, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,    4,   60,    4,    32,   960,   411>,
        block_fft_implementation<60,    4,   60,    4,    16,  1920,   414>,
        block_fft_implementation<30,    2,   30,    8,     4,  1920,   413>,
        block_fft_implementation<30,    2,   30,    8,    16,   960,   412>
    >;
};

template<> struct block_fft_record<248, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    4,     8,   992,   415>,
        block_fft_implementation<62,    2,   62,    4,     8,  1984,   416>
    >;
};

template<> struct block_fft_record<252, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    6,   42,    6,    64,  1008,   417>,
        block_fft_implementation<42,    6,   42,    6,    21,  2016,   418>
    >;
};

template<> struct block_fft_record<272, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    8,     8,  1088,   420>,
        block_fft_implementation<34,    2,   34,    8,     4,  2176,   419>
    >;
};

template<> struct block_fft_record<288, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   12,   24,   12,     5,  1152,   421>,
        block_fft_implementation<24,   12,   24,   12,     5,  2304,   423>,
        block_fft_implementation<12,    2,   12,   24,     5,  1152,   422>,
        block_fft_implementation<12,    2,   12,   24,     1,  2304,   424>,
        block_fft_implementation<36,    2,   36,    8,     4,  2304,   426>,
        block_fft_implementation<48,    6,   48,    6,     5,  1152,   425>,
        block_fft_implementation<18,    2,   18,   16,     4,  2304,   427>,
        block_fft_implementation< 6,    2,    6,   48,     4,  2304,   428>,
        block_fft_implementation<48,    6,   48,    6,    10,  2304,   430>,
        block_fft_implementation<36,    2,   36,    8,    16,  1152,   429>,
        block_fft_implementation< 6,    2,    6,   48,     8,  1152,   431>,
        block_fft_implementation<18,    2,   18,   16,    16,  1152,   432>
    >;
};

template<> struct block_fft_record<300, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   10,   30,   10,     6,  1200,   433>,
        block_fft_implementation<30,   10,   30,   10,     6,  2400,   434>,
        block_fft_implementation<60,    5,   60,    5,     6,  1200,   435>,
        block_fft_implementation<60,    5,   60,    5,     6,  2400,   436>
    >;
};

template<> struct block_fft_record<304, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    8,     8,  1216,   438>,
        block_fft_implementation<38,    2,   38,    8,     8,  2432,   437>
    >;
};

template<> struct block_fft_record<320, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<40,    8,   40,    8,     4,  1280,   439>,
        block_fft_implementation<20,    4,   20,   16,     2,  2560,   440>,
        block_fft_implementation<40,    8,   40,    8,     4,  2560,   442>,
        block_fft_implementation<20,    4,   20,   16,     8,  1280,   441>,
        block_fft_implementation<10,    2,   10,   32,     2,  2560,   443>,
        block_fft_implementation<10,    2,   10,   32,     4,  1280,   444>
    >;
};

template<> struct block_fft_record<336, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    8,     4,  1344,   446>,
        block_fft_implementation<42,    2,   42,    8,     4,  2688,   445>
    >;
};

template<> struct block_fft_record<352, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,    2,   44,    8,     8,  2816,   447>,
        block_fft_implementation<22,    2,   22,   16,     2,  2816,   448>,
        block_fft_implementation<44,    2,   44,    8,     4,  1408,   449>,
        block_fft_implementation<22,    2,   22,   16,    16,  1408,   450>
    >;
};

template<> struct block_fft_record<360, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,    6,   60,    6,     5,  1440,   451>,
        block_fft_implementation<30,    2,   30,   12,    21,  1440,   452>,
        block_fft_implementation<30,    2,   30,   12,     5,  2880,   453>,
        block_fft_implementation<60,    6,   60,    6,     5,  2880,   454>
    >;
};

template<> struct block_fft_record<368, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    8,     4,  1472,   456>,
        block_fft_implementation<46,    2,   46,    8,     4,  2944,   455>
    >;
};

template<> struct block_fft_record<384, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<48,    8,   48,    8,     4,  1536,   458>,
        block_fft_implementation<24,    2,   24,   16,     8,  1536,   459>,
        block_fft_implementation<24,    2,   24,   16,     2,  3072,   461>,
        block_fft_implementation<12,    2,   12,   32,     2,  1536,   457>,
        block_fft_implementation<12,    2,   12,   32,     1,  3072,   460>,
        block_fft_implementation<48,    8,   48,    8,     8,  3072,   462>,
        block_fft_implementation< 6,    2,    6,   64,     4,  3072,   463>,
        block_fft_implementation< 6,    2,    6,   64,     4,  1536,   464>
    >;
};

template<> struct block_fft_record<392, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   14,   28,   14,     9,  1568,   465>,
        block_fft_implementation<28,   14,   28,   14,     9,  3136,   466>,
        block_fft_implementation<14,    2,   14,   28,     1,  3136,   469>,
        block_fft_implementation<56,    7,   56,    7,     9,  1568,   467>,
        block_fft_implementation<14,    2,   14,   28,     1,  1568,   468>,
        block_fft_implementation<56,    7,   56,    7,     9,  3136,   470>
    >;
};

template<> struct block_fft_record<416, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,    2,   52,    8,     8,  3328,   472>,
        block_fft_implementation<26,    2,   26,   16,     4,  3328,   471>,
        block_fft_implementation<52,    2,   52,    8,     4,  1664,   473>,
        block_fft_implementation<26,    2,   26,   16,     4,  1664,   474>
    >;
};

template<> struct block_fft_record<432, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<36,   12,   36,   12,    10,  1728,   475>,
        block_fft_implementation<36,   12,   36,   12,     9,  3456,   476>,
        block_fft_implementation<12,    3,   12,   36,     7,  1728,   477>,
        block_fft_implementation<12,    3,   12,   36,     7,  3456,   479>,
        block_fft_implementation<24,    3,   24,   18,     7,  1728,   478>,
        block_fft_implementation< 6,    2,    6,   72,     3,  3456,   481>,
        block_fft_implementation<24,    3,   24,   18,     3,  3456,   482>,
        block_fft_implementation<18,    2,   18,   24,     1,  3456,   483>,
        block_fft_implementation< 6,    2,    6,   72,     3,  1728,   480>,
        block_fft_implementation<48,    3,   48,    9,     7,  1728,   485>,
        block_fft_implementation<18,    2,   18,   24,     1,  1728,   484>,
        block_fft_implementation<54,    2,   54,    8,    16,  1728,   486>,
        block_fft_implementation<48,    3,   48,    9,     6,  3456,   487>,
        block_fft_implementation<54,    2,   54,    8,     4,  3456,   488>
    >;
};

template<> struct block_fft_record<448, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<56,    8,   56,    8,     8,  1792,   490>,
        block_fft_implementation<28,    4,   28,   16,     4,  3584,   489>,
        block_fft_implementation<56,    8,   56,    8,     4,  3584,   492>,
        block_fft_implementation<28,    4,   28,   16,     4,  1792,   491>,
        block_fft_implementation<14,    2,   14,   32,     3,  3584,   493>,
        block_fft_implementation<14,    2,   14,   32,     2,  1792,   494>
    >;
};

template<> struct block_fft_record<464, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    8,    12,  1856,   495>,
        block_fft_implementation<58,    2,   58,    8,     4,  3712,   496>
    >;
};

template<> struct block_fft_record<480, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,    2,   30,   16,     2,  3840,   497>,
        block_fft_implementation<60,    2,   60,    8,     4,  1920,   498>,
        block_fft_implementation<60,    2,   60,    8,    15,  3840,   500>,
        block_fft_implementation<30,    2,   30,   16,     4,  1920,   499>
    >;
};

template<> struct block_fft_record<496, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    8,     8,  1984,   501>,
        block_fft_implementation<62,    2,   62,    8,     4,  3968,   502>
    >;
};

template<> struct block_fft_record<500, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<50,   10,   50,   10,    25,  2000,   504>,
        block_fft_implementation<50,   10,   50,   10,     6,  4000,   506>,
        block_fft_implementation<20,    5,   20,   25,     5,  2000,   503>,
        block_fft_implementation<10,    5,   10,   50,     1,  4000,   508>,
        block_fft_implementation<20,    5,   20,   25,     1,  4000,   507>,
        block_fft_implementation<10,    5,   10,   50,     5,  2000,   505>
    >;
};

template<> struct block_fft_record<504, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,   12,    32,  2016,   510>,
        block_fft_implementation<42,    2,   42,   12,     5,  4032,   509>
    >;
};
