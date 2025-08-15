//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<33, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<33,   33,   33,    1,    64,     0,   974>
    >;
};

template<> struct block_fft_record<34, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,   34,   34,    1,    64,     0,   975>
    >;
};

template<> struct block_fft_record<35, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<35,   35,   35,    1,    64,     0,   976>
    >;
};

template<> struct block_fft_record<37, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<37,   37,   37,    1,    64,     0,   977>
    >;
};

template<> struct block_fft_record<38, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,   38,   38,    1,    64,     0,   978>
    >;
};

template<> struct block_fft_record<39, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<39,   39,   39,    1,    64,     0,   979>
    >;
};

template<> struct block_fft_record<40, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,    2,   20,    2,    32,    80,   980>,
        block_fft_implementation<20,    2,   20,    2,    32,   160,   981>,
        block_fft_implementation<10,    2,   10,    4,    64,   160,   982>,
        block_fft_implementation<40,   40,   40,    1,    32,     0,   983>
    >;
};

template<> struct block_fft_record<41, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<41,   41,   41,    1,    64,     0,   984>
    >;
};

template<> struct block_fft_record<42, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,   42,   42,    1,    64,     0,   985>
    >;
};

template<> struct block_fft_record<43, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<43,   43,   43,    1,    64,     0,   986>
    >;
};

template<> struct block_fft_record<44, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,   44,   44,    1,    64,     0,   987>
    >;
};

template<> struct block_fft_record<45, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<45,   45,   45,    1,    64,     0,   988>
    >;
};

template<> struct block_fft_record<46, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,   46,   46,    1,    64,     0,   989>
    >;
};

template<> struct block_fft_record<47, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<47,   47,   47,    1,    64,     0,   990>
    >;
};

template<> struct block_fft_record<48, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<48,   48,   48,    1,    64,     0,   991>
    >;
};

template<> struct block_fft_record<50, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<50,   50,   50,    1,    64,     0,   992>
    >;
};

template<> struct block_fft_record<51, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<51,   51,   51,    1,    64,     0,   993>
    >;
};

template<> struct block_fft_record<52, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,   52,   52,    1,    64,     0,   994>
    >;
};

template<> struct block_fft_record<53, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<53,   53,   53,    1,    64,     0,   995>
    >;
};

template<> struct block_fft_record<54, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<54,   54,   54,    1,    64,     0,   996>
    >;
};

template<> struct block_fft_record<55, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<55,   55,   55,    1,    64,     0,   997>
    >;
};

template<> struct block_fft_record<56, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<56,   56,   56,    1,    64,     0,   998>
    >;
};

template<> struct block_fft_record<57, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<57,   57,   57,    1,    64,     0,   999>
    >;
};

template<> struct block_fft_record<58, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,   58,   58,    1,    64,     0,  1000>
    >;
};

template<> struct block_fft_record<59, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<59,   59,   59,    1,    64,     0,  1001>
    >;
};

template<> struct block_fft_record<60, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,   60,   60,    1,    64,     0,  1002>
    >;
};

template<> struct block_fft_record<61, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<61,   61,   61,    1,    64,     0,  1003>
    >;
};

template<> struct block_fft_record<62, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,   62,   62,    1,    64,     0,  1004>
    >;
};

template<> struct block_fft_record<63, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<63,   63,   63,    1,    64,     0,  1005>
    >;
};

template<> struct block_fft_record<2, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    2,    1,   128,     0,  1006>
    >;
};

template<> struct block_fft_record<4, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    4,    1,   128,     0,  1007>,
        block_fft_implementation< 2,    2,    2,    2,   256,     8,  1008>,
        block_fft_implementation< 2,    2,    2,    2,   256,    16,  1009>
    >;
};

template<> struct block_fft_record<8, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    1,    64,     0,  1010>,
        block_fft_implementation< 4,    2,    4,    2,   256,    16,  1011>,
        block_fft_implementation< 4,    2,    4,    2,   192,    32,  1012>,
        block_fft_implementation< 2,    2,    2,    4,   160,    16,  1013>,
        block_fft_implementation< 2,    2,    2,    4,   256,    32,  1014>
    >;
};

template<> struct block_fft_record<16, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,    1,   224,     0,  1015>,
        block_fft_implementation< 4,    4,    4,    4,   128,    32,  1016>,
        block_fft_implementation< 4,    4,    4,    4,   192,    64,  1017>,
        block_fft_implementation< 8,    2,    8,    2,   256,    32,  1018>,
        block_fft_implementation< 8,    2,    8,    2,   224,    64,  1019>,
        block_fft_implementation< 2,    2,    2,    8,   128,    32,  1020>,
        block_fft_implementation< 2,    2,    2,    8,   128,    64,  1021>
    >;
};

template<> struct block_fft_record<32, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<32,   32,   32,    1,    64,     0,  1022>,
        block_fft_implementation< 8,    4,    8,    4,    96,   128,  1023>,
        block_fft_implementation< 8,    4,    8,    4,   192,    64,  1024>,
        block_fft_implementation< 4,    2,    4,    8,    96,   128,  1025>,
        block_fft_implementation< 4,    2,    4,    8,   128,    64,  1026>,
        block_fft_implementation< 2,    2,    2,   16,    40,   128,  1027>,
        block_fft_implementation<16,    2,   16,    2,   128,   128,  1028>,
        block_fft_implementation< 2,    2,    2,   16,    52,    64,  1029>,
        block_fft_implementation<16,    2,   16,    2,   256,    64,  1030>
    >;
};

template<> struct block_fft_record<64, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    8,    48,   256,  1031>,
        block_fft_implementation< 8,    8,    8,    8,    26,   128,  1032>,
        block_fft_implementation< 4,    4,    4,   16,    18,   256,  1033>,
        block_fft_implementation< 4,    4,    4,   16,    30,   128,  1034>,
        block_fft_implementation<16,    4,   16,    4,    50,   256,  1035>,
        block_fft_implementation<16,    4,   16,    4,    58,   128,  1036>,
        block_fft_implementation<32,    2,   32,    2,    46,   256,  1037>,
        block_fft_implementation< 2,    2,    2,   32,    28,   256,  1038>,
        block_fft_implementation< 2,    2,    2,   32,    16,   128,  1039>,
        block_fft_implementation<32,    2,   32,    2,   128,   128,  1040>,
        block_fft_implementation<64,   64,   64,    1,    64,     0,  1041>
    >;
};

template<> struct block_fft_record<128, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    8,   16,    8,    32,   256,  1042>,
        block_fft_implementation< 8,    2,    8,   16,    16,   512,  1043>,
        block_fft_implementation<16,    8,   16,    8,    26,   512,  1044>,
        block_fft_implementation< 8,    2,    8,   16,    32,   256,  1045>,
        block_fft_implementation< 4,    2,    4,   32,     8,   512,  1046>,
        block_fft_implementation< 4,    2,    4,   32,    14,   256,  1047>,
        block_fft_implementation<32,    4,   32,    4,     8,   512,  1048>,
        block_fft_implementation<32,    4,   32,    4,    62,   256,  1049>,
        block_fft_implementation< 2,    2,    2,   64,    12,   512,  1050>,
        block_fft_implementation< 2,    2,    2,   64,    12,   256,  1051>
    >;
};

template<> struct block_fft_record<256, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,   16,     4,   512,  1052>,
        block_fft_implementation< 8,    4,    8,   32,     8,  1024,  1053>,
        block_fft_implementation< 8,    4,    8,   32,     8,   512,  1054>,
        block_fft_implementation<16,   16,   16,   16,     6,  1024,  1055>,
        block_fft_implementation<32,    8,   32,    8,     8,  1024,  1056>,
        block_fft_implementation< 4,    4,    4,   64,     4,  1024,  1057>,
        block_fft_implementation< 4,    4,    4,   64,     8,   512,  1058>,
        block_fft_implementation<32,    8,   32,    8,    32,   512,  1059>,
        block_fft_implementation< 2,    2,    2,  128,     6,  1024,  1060>,
        block_fft_implementation< 2,    2,    2,  128,     8,   512,  1061>
    >;
};

template<> struct block_fft_record<512, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,   64,     4,  2048,  1062>,
        block_fft_implementation< 8,    8,    8,   64,     4,  1024,  1063>,
        block_fft_implementation<16,    2,   16,   32,     2,  1024,  1064>,
        block_fft_implementation<32,   16,   32,   16,     4,  2048,  1065>,
        block_fft_implementation<32,   16,   32,   16,     4,  1024,  1066>,
        block_fft_implementation<16,    2,   16,   32,     8,  2048,  1067>,
        block_fft_implementation< 4,    2,    4,  128,     4,  2048,  1068>,
        block_fft_implementation< 4,    2,    4,  128,     4,  1024,  1069>,
        block_fft_implementation< 2,    2,    2,  256,     2,  2048,  1070>,
        block_fft_implementation< 2,    2,    2,  256,     4,  1024,  1071>
    >;
};

template<> struct block_fft_record<1024, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    4,   16,   64,     2,  2048,  1072>,
        block_fft_implementation< 8,    2,    8,  128,     2,  4096,  1073>,
        block_fft_implementation<32,   32,   32,   32,     8,  4096,  1074>,
        block_fft_implementation< 8,    2,    8,  128,     2,  2048,  1075>,
        block_fft_implementation<32,   32,   32,   32,     8,  2048,  1076>,
        block_fft_implementation<16,    4,   16,   64,     4,  4096,  1077>,
        block_fft_implementation< 4,    4,    4,  256,     2,  4096,  1078>,
        block_fft_implementation< 4,    4,    4,  256,     2,  2048,  1079>,
        block_fft_implementation< 2,    2,    2,  512,     2,  4096,  1080>,
        block_fft_implementation< 2,    2,    2,  512,     2,  2048,  1081>
    >;
};

template<> struct block_fft_record<2048, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    8,   16,  128,     2,  4096,  1082>,
        block_fft_implementation< 8,    4,    8,  256,     2,  8192,  1083>,
        block_fft_implementation< 8,    4,    8,  256,     2,  4096,  1084>,
        block_fft_implementation<16,    8,   16,  128,     2,  8192,  1085>,
        block_fft_implementation<32,    2,   32,   64,     4,  8192,  1086>,
        block_fft_implementation<32,    2,   32,   64,     4,  4096,  1087>,
        block_fft_implementation< 4,    2,    4,  512,     2,  8192,  1088>,
        block_fft_implementation< 4,    2,    4,  512,     2,  4096,  1089>,
        block_fft_implementation< 2,    2,    2, 1024,     2,  8192,  1090>,
        block_fft_implementation< 2,    2,    2, 1024,     2,  4096,  1091>
    >;
};

template<> struct block_fft_record<4096, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,  256,     2,  8192,  1092>,
        block_fft_implementation< 8,    8,    8,  512,     2, 16384,  1093>,
        block_fft_implementation< 8,    8,    8,  512,     2,  8192,  1094>,
        block_fft_implementation<16,   16,   16,  256,     2, 16384,  1095>,
        block_fft_implementation<32,    4,   32,  128,     2, 16384,  1096>,
        block_fft_implementation<32,    4,   32,  128,     2,  8192,  1097>,
        block_fft_implementation< 4,    4,    4, 1024,     2, 16384,  1098>,
        block_fft_implementation< 4,    4,    4, 1024,     2,  8192,  1099>
    >;
};

template<> struct block_fft_record<8192, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    2,   16,  512,     2, 16384,  1100>,
        block_fft_implementation<32,    8,   32,  256,     2, 32768,  1101>,
        block_fft_implementation<32,    8,   32,  256,     2, 16384,  1102>,
        block_fft_implementation< 8,    2,    8, 1024,     2, 32768,  1103>,
        block_fft_implementation<16,    2,   16,  512,     2, 32768,  1104>,
        block_fft_implementation< 8,    2,    8, 1024,     2, 16384,  1105>
    >;
};

template<> struct block_fft_record<16384, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,    4,   16, 1024,     2, 32768,  1106>,
        block_fft_implementation<32,   16,   32,  512,     2, 32768,  1107>
    >;
};

template<> struct block_fft_record<3, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    3,    1,   128,     0,  1108>
    >;
};

template<> struct block_fft_record<9, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    1,   128,     0,  1109>,
        block_fft_implementation< 3,    3,    3,    3,   128,    36,  1110>,
        block_fft_implementation< 3,    3,    3,    3,   256,    18,  1111>
    >;
};

template<> struct block_fft_record<27, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,    1,    64,     0,  1112>,
        block_fft_implementation< 9,    3,    9,    3,    42,    54,  1113>,
        block_fft_implementation< 9,    3,    9,    3,    20,   108,  1114>,
        block_fft_implementation< 3,    3,    3,    9,    52,   108,  1115>,
        block_fft_implementation< 3,    3,    3,    9,    48,    54,  1116>
    >;
};

template<> struct block_fft_record<81, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    9,    28,   324,  1117>,
        block_fft_implementation< 9,    9,    9,    9,    14,   162,  1118>,
        block_fft_implementation<27,    3,   27,    3,    20,   162,  1119>,
        block_fft_implementation<27,    3,   27,    3,    18,   324,  1120>,
        block_fft_implementation< 3,    3,    3,   27,    14,   324,  1121>,
        block_fft_implementation< 3,    3,    3,   27,    18,   162,  1122>
    >;
};

template<> struct block_fft_record<243, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,    9,   27,    9,    28,   972,  1123>,
        block_fft_implementation<27,    9,   27,    9,    28,   486,  1124>,
        block_fft_implementation< 9,    3,    9,   27,    18,   972,  1125>,
        block_fft_implementation< 9,    3,    9,   27,    18,   486,  1126>,
        block_fft_implementation< 3,    3,    3,   81,     6,   972,  1127>,
        block_fft_implementation< 3,    3,    3,   81,     6,   486,  1128>
    >;
};

template<> struct block_fft_record<729, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,   27,    28,  2916,  1129>,
        block_fft_implementation<27,   27,   27,   27,    18,  1458,  1130>,
        block_fft_implementation< 9,    9,    9,   81,     6,  2916,  1131>,
        block_fft_implementation< 9,    9,    9,   81,     6,  1458,  1132>,
        block_fft_implementation< 3,    3,    3,  243,     2,  2916,  1133>,
        block_fft_implementation< 3,    3,    3,  243,     2,  1458,  1134>
    >;
};

template<> struct block_fft_record<2187, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    3,    9,  243,     2,  8748,  1135>,
        block_fft_implementation<27,    3,   27,   81,     6,  8748,  1136>,
        block_fft_implementation< 9,    3,    9,  243,     2,  4374,  1137>,
        block_fft_implementation<27,    3,   27,   81,     6,  4374,  1138>,
        block_fft_implementation< 3,    3,    3,  729,     2,  8748,  1139>,
        block_fft_implementation< 3,    3,    3,  729,     2,  4374,  1140>
    >;
};

template<> struct block_fft_record<6561, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,    9,   27,  243,     2, 13122,  1141>,
        block_fft_implementation<27,    9,   27,  243,     2, 26244,  1142>,
        block_fft_implementation< 9,    9,    9,  729,     2, 13122,  1143>,
        block_fft_implementation< 9,    9,    9,  729,     2, 26244,  1144>
    >;
};

template<> struct block_fft_record<19683, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,  729,     2, 39366,  1145>
    >;
};

template<> struct block_fft_record<5, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    5,    1,    64,     0,  1146>
    >;
};

template<> struct block_fft_record<25, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,    1,    64,     0,  1147>,
        block_fft_implementation< 5,    5,    5,    5,   256,   100,  1148>,
        block_fft_implementation< 5,    5,    5,    5,    50,    50,  1149>
    >;
};

template<> struct block_fft_record<125, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,    5,   25,    5,    50,   250,  1150>,
        block_fft_implementation<25,    5,   25,    5,    50,   500,  1151>,
        block_fft_implementation< 5,    5,    5,   25,    10,   500,  1152>,
        block_fft_implementation< 5,    5,    5,   25,    10,   250,  1153>
    >;
};

template<> struct block_fft_record<625, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,   25,    10,  1250,  1154>,
        block_fft_implementation<25,   25,   25,   25,    10,  2500,  1155>,
        block_fft_implementation< 5,    5,    5,  125,     2,  2500,  1156>,
        block_fft_implementation< 5,    5,    5,  125,     2,  1250,  1157>
    >;
};

template<> struct block_fft_record<3125, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,    5,   25,  125,     2,  6250,  1158>,
        block_fft_implementation<25,    5,   25,  125,     6, 12500,  1159>,
        block_fft_implementation< 5,    5,    5,  625,     2, 12500,  1160>,
        block_fft_implementation< 5,    5,    5,  625,     2,  6250,  1161>
    >;
};

template<> struct block_fft_record<15625, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,  625,     2, 31250,  1162>
    >;
};

template<> struct block_fft_record<7, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,    1,   128,     0,  1163>
    >;
};

template<> struct block_fft_record<49, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<49,   49,   49,    1,    64,     0,  1164>,
        block_fft_implementation< 7,    7,    7,    7,    18,   196,  1165>,
        block_fft_implementation< 7,    7,    7,    7,    18,    98,  1166>
    >;
};

template<> struct block_fft_record<343, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,   49,    10,  1372,  1167>,
        block_fft_implementation< 7,    7,    7,   49,    10,   686,  1168>
    >;
};

template<> struct block_fft_record<2401, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,  343,     2,  9604,  1169>,
        block_fft_implementation< 7,    7,    7,  343,     2,  4802,  1170>
    >;
};

template<> struct block_fft_record<16807, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<49,    7,   49,  343,     2, 33614,  1171>
    >;
};

template<> struct block_fft_record<11, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    1,   192,     0,  1172>
    >;
};

template<> struct block_fft_record<121, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,   11,    46,   484,  1173>,
        block_fft_implementation<11,   11,   11,   11,    46,   242,  1174>
    >;
};

template<> struct block_fft_record<1331, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,  121,     2,  2662,  1175>,
        block_fft_implementation<11,   11,   11,  121,     2,  5324,  1176>
    >;
};

template<> struct block_fft_record<6, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    1,    64,     0,  1177>
    >;
};

template<> struct block_fft_record<36, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    6,   128,    72,  1178>,
        block_fft_implementation< 6,    6,    6,    6,   128,   144,  1179>,
        block_fft_implementation<36,   36,   36,    1,    72,     0,  1180>
    >;
};

template<> struct block_fft_record<216, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,   36,    14,   864,  1181>,
        block_fft_implementation< 6,    6,    6,   36,    14,   432,  1182>
    >;
};

template<> struct block_fft_record<1296, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<36,   36,   36,   36,    14,  2592,  1183>,
        block_fft_implementation<36,   36,   36,   36,     6,  5184,  1184>,
        block_fft_implementation< 6,    6,    6,  216,     2,  5184,  1185>,
        block_fft_implementation< 6,    6,    6,  216,     2,  2592,  1186>
    >;
};

template<> struct block_fft_record<7776, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<36,    6,   36,  216,     2, 31104,  1187>,
        block_fft_implementation<36,    6,   36,  216,     2, 15552,  1188>
    >;
};

template<> struct block_fft_record<10, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,    1,    64,     0,  1189>
    >;
};

template<> struct block_fft_record<100, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,   10,    50,   200,  1190>,
        block_fft_implementation<10,   10,   10,   10,    50,   400,  1191>
    >;
};

template<> struct block_fft_record<1000, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,  100,    10,  4000,  1192>,
        block_fft_implementation<10,   10,   10,  100,    10,  2000,  1193>
    >;
};

template<> struct block_fft_record<10000, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10, 1000,     2, 40000,  1194>,
        block_fft_implementation<10,   10,   10, 1000,     2, 20000,  1195>
    >;
};

template<> struct block_fft_record<12, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,    1,   128,     0,  1196>
    >;
};

template<> struct block_fft_record<144, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,   12,    42,   576,  1197>,
        block_fft_implementation<12,   12,   12,   12,    42,   288,  1198>
    >;
};

template<> struct block_fft_record<1728, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,  144,     2,  6912,  1199>,
        block_fft_implementation<12,   12,   12,  144,     2,  3456,  1200>
    >;
};

template<> struct block_fft_record<13, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,    1,    64,     0,  1201>
    >;
};

template<> struct block_fft_record<169, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,   13,    38,   676,  1202>,
        block_fft_implementation<13,   13,   13,   13,    38,   338,  1203>
    >;
};

template<> struct block_fft_record<2197, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,   13,  169,     6,  4394,  1204>,
        block_fft_implementation<13,   13,   13,  169,     4,  8788,  1205>
    >;
};

template<> struct block_fft_record<14, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,    1,    64,     0,  1206>
    >;
};

template<> struct block_fft_record<196, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,   14,    18,   784,  1207>,
        block_fft_implementation<14,   14,   14,   14,    18,   392,  1208>
    >;
};

template<> struct block_fft_record<2744, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,   14,  196,     4, 10976,  1209>,
        block_fft_implementation<14,   14,   14,  196,     2,  5488,  1210>
    >;
};

template<> struct block_fft_record<15, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,    1,    64,     0,  1211>
    >;
};

template<> struct block_fft_record<225, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,   15,    34,   900,  1212>,
        block_fft_implementation<15,   15,   15,   15,    34,   450,  1213>
    >;
};

template<> struct block_fft_record<3375, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,   15,  225,     2, 13500,  1214>,
        block_fft_implementation<15,   15,   15,  225,     2,  6750,  1215>
    >;
};

template<> struct block_fft_record<17, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,    1,    64,     0,  1216>
    >;
};

template<> struct block_fft_record<289, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,   17,    30,  1156,  1217>,
        block_fft_implementation<17,   17,   17,   17,    30,   578,  1218>
    >;
};

template<> struct block_fft_record<4913, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,   17,  289,     4, 19652,  1219>,
        block_fft_implementation<17,   17,   17,  289,     4,  9826,  1220>
    >;
};

template<> struct block_fft_record<18, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,    1,    64,     0,  1221>
    >;
};

template<> struct block_fft_record<324, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,   18,    14,  1296,  1222>,
        block_fft_implementation<18,   18,   18,   18,    14,   648,  1223>
    >;
};

template<> struct block_fft_record<5832, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,   18,  324,     2, 23328,  1224>,
        block_fft_implementation<18,   18,   18,  324,     2, 11664,  1225>
    >;
};

template<> struct block_fft_record<19, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,    1,    64,     0,  1226>
    >;
};

template<> struct block_fft_record<361, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,   19,    26,  1444,  1227>,
        block_fft_implementation<19,   19,   19,   19,    26,   722,  1228>
    >;
};

template<> struct block_fft_record<6859, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,   19,  361,     2, 27436,  1229>,
        block_fft_implementation<19,   19,   19,  361,     2, 13718,  1230>
    >;
};

template<> struct block_fft_record<20, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,    1,    64,     0,  1231>
    >;
};

template<> struct block_fft_record<400, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,   20,    12,  1600,  1232>,
        block_fft_implementation<20,   20,   20,   20,    12,   800,  1233>
    >;
};

template<> struct block_fft_record<8000, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,   20,  400,     2, 16000,  1234>,
        block_fft_implementation<20,   20,   20,  400,     2, 32000,  1235>
    >;
};

template<> struct block_fft_record<21, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,   21,    1,    64,     0,  1236>
    >;
};

template<> struct block_fft_record<441, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,   21,   21,    12,  1764,  1237>,
        block_fft_implementation<21,   21,   21,   21,    12,   882,  1238>
    >;
};

template<> struct block_fft_record<9261, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,   21,  441,     2, 18522,  1239>,
        block_fft_implementation<21,   21,   21,  441,     2, 37044,  1240>
    >;
};

template<> struct block_fft_record<22, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,   22,    1,    64,     0,  1241>
    >;
};

template<> struct block_fft_record<484, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,   22,   22,    22,  1936,  1242>,
        block_fft_implementation<22,   22,   22,   22,    22,   968,  1243>
    >;
};

template<> struct block_fft_record<10648, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,   22,  484,     2, 42592,  1244>,
        block_fft_implementation<22,   22,   22,  484,     2, 21296,  1245>
    >;
};

template<> struct block_fft_record<23, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,   23,    1,    64,     0,  1246>
    >;
};

template<> struct block_fft_record<529, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,   23,   23,    22,  2116,  1247>,
        block_fft_implementation<23,   23,   23,   23,    22,  1058,  1248>
    >;
};

template<> struct block_fft_record<12167, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,   23,  529,     2, 48668,  1249>,
        block_fft_implementation<23,   23,   23,  529,     2, 24334,  1250>
    >;
};

template<> struct block_fft_record<24, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,   24,    1,    64,     0,  1251>
    >;
};

template<> struct block_fft_record<576, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,   24,   24,    10,  1152,  1252>,
        block_fft_implementation<24,   24,   24,   24,    10,  2304,  1253>
    >;
};

template<> struct block_fft_record<13824, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,   24,  576,     2, 27648,  1254>
    >;
};

template<> struct block_fft_record<26, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,   26,    1,    64,     0,  1255>
    >;
};

template<> struct block_fft_record<676, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,   26,   26,    28,  2704,  1256>,
        block_fft_implementation<26,   26,   26,   26,    18,  1352,  1257>
    >;
};

template<> struct block_fft_record<17576, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,   26,  676,     2, 35152,  1258>
    >;
};

template<> struct block_fft_record<28, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,   28,    1,    64,     0,  1259>
    >;
};

template<> struct block_fft_record<784, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,   28,   28,    18,  1568,  1260>,
        block_fft_implementation<28,   28,   28,   28,    18,  3136,  1261>
    >;
};

template<> struct block_fft_record<21952, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,   28,  784,     2, 43904,  1262>
    >;
};

template<> struct block_fft_record<29, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,   29,    1,    64,     0,  1263>
    >;
};

template<> struct block_fft_record<841, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,   29,   29,    26,  3364,  1264>,
        block_fft_implementation<29,   29,   29,   29,    26,  1682,  1265>
    >;
};

template<> struct block_fft_record<24389, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,   29,  841,     2, 48778,  1266>
    >;
};

template<> struct block_fft_record<30, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,   30,    1,    64,     0,  1267>
    >;
};

template<> struct block_fft_record<900, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,   30,   30,     8,  3600,  1268>,
        block_fft_implementation<30,   30,   30,   30,     8,  1800,  1269>
    >;
};

template<> struct block_fft_record<31, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,   31,    1,    64,     0,  1270>
    >;
};

template<> struct block_fft_record<961, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,   31,   31,    16,  3844,  1271>,
        block_fft_implementation<31,   31,   31,   31,    16,  1922,  1272>
    >;
};

template<> struct block_fft_record<68, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    2,   128,   136,  1273>,
        block_fft_implementation<34,    2,   34,    2,    30,   272,  1274>
    >;
};

template<> struct block_fft_record<72, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,    6,   12,    6,    42,   144,  1275>,
        block_fft_implementation<12,    6,   12,    6,    42,   288,  1276>,
        block_fft_implementation<24,    3,   24,    3,    20,   144,  1277>,
        block_fft_implementation<36,    2,   36,    2,    32,   144,  1278>,
        block_fft_implementation<24,    3,   24,    3,    20,   288,  1279>,
        block_fft_implementation< 6,    2,    6,   12,    20,   288,  1280>,
        block_fft_implementation<18,    2,   18,    4,    16,   144,  1281>,
        block_fft_implementation<18,    2,   18,    4,    64,   288,  1282>,
        block_fft_implementation< 6,    2,    6,   12,    16,   144,  1283>,
        block_fft_implementation<36,    2,   36,    2,    28,   288,  1284>
    >;
};

template<> struct block_fft_record<76, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    2,   128,   152,  1285>,
        block_fft_implementation<38,    2,   38,    2,   128,   304,  1286>
    >;
};

template<> struct block_fft_record<80, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,    4,   20,    4,    16,   160,  1287>,
        block_fft_implementation<20,    4,   20,    4,    16,   320,  1288>,
        block_fft_implementation<40,    2,   40,    2,    32,   160,  1289>,
        block_fft_implementation<10,    2,   10,    8,    32,   320,  1290>,
        block_fft_implementation<10,    2,   10,    8,    32,   160,  1291>,
        block_fft_implementation<40,    2,   40,    2,    24,   320,  1292>
    >;
};

template<> struct block_fft_record<84, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    2,   128,   168,  1293>,
        block_fft_implementation<42,    2,   42,    2,   128,   336,  1294>
    >;
};

template<> struct block_fft_record<88, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,    2,   44,    2,   128,   176,  1295>,
        block_fft_implementation<22,    2,   22,    4,    64,   176,  1296>,
        block_fft_implementation<22,    2,   22,    4,    32,   352,  1297>,
        block_fft_implementation<44,    2,   44,    2,   128,   352,  1298>
    >;
};

template<> struct block_fft_record<92, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    2,   128,   184,  1299>,
        block_fft_implementation<46,    2,   46,    2,    32,   368,  1300>
    >;
};

template<> struct block_fft_record<96, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,    4,   24,    4,    16,   384,  1301>,
        block_fft_implementation<12,    2,   12,    8,    32,   384,  1302>,
        block_fft_implementation<24,    4,   24,    4,    62,   192,  1303>,
        block_fft_implementation< 6,    2,    6,   16,    16,   384,  1304>,
        block_fft_implementation<12,    2,   12,    8,    32,   192,  1305>,
        block_fft_implementation< 6,    2,    6,   16,    30,   192,  1306>,
        block_fft_implementation<48,    2,   48,    2,    32,   384,  1307>,
        block_fft_implementation<48,    2,   48,    2,    30,   192,  1308>
    >;
};

template<> struct block_fft_record<104, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,    2,   52,    2,   128,   208,  1309>,
        block_fft_implementation<26,    2,   26,    4,    64,   208,  1310>,
        block_fft_implementation<26,    2,   26,    4,    64,   416,  1311>,
        block_fft_implementation<52,    2,   52,    2,    28,   416,  1312>
    >;
};

template<> struct block_fft_record<108, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,    6,   18,    6,    42,   432,  1313>,
        block_fft_implementation<18,    6,   18,    6,    42,   216,  1314>,
        block_fft_implementation<36,    3,   36,    3,    20,   216,  1315>,
        block_fft_implementation<54,    2,   54,    2,   128,   216,  1316>,
        block_fft_implementation< 6,    3,    6,   18,    14,   432,  1317>,
        block_fft_implementation< 6,    3,    6,   18,    14,   216,  1318>,
        block_fft_implementation<36,    3,   36,    3,    20,   432,  1319>,
        block_fft_implementation<12,    3,   12,    9,    28,   216,  1320>,
        block_fft_implementation<12,    3,   12,    9,    28,   432,  1321>,
        block_fft_implementation<54,    2,   54,    2,    28,   432,  1322>
    >;
};

template<> struct block_fft_record<112, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,    4,   28,    4,    16,   224,  1323>,
        block_fft_implementation<28,    4,   28,    4,    64,   448,  1324>,
        block_fft_implementation<56,    2,   56,    2,   128,   224,  1325>,
        block_fft_implementation<14,    2,   14,    8,    32,   448,  1326>,
        block_fft_implementation<14,    2,   14,    8,    32,   224,  1327>,
        block_fft_implementation<56,    2,   56,    2,    64,   448,  1328>
    >;
};

template<> struct block_fft_record<116, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    2,    32,   232,  1329>,
        block_fft_implementation<58,    2,   58,    2,    26,   464,  1330>
    >;
};

template<> struct block_fft_record<120, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,    2,   60,    2,   128,   240,  1331>,
        block_fft_implementation<30,    2,   30,    4,    64,   240,  1332>,
        block_fft_implementation<30,    2,   30,    4,    16,   480,  1333>,
        block_fft_implementation<60,    2,   60,    2,    96,   480,  1334>
    >;
};

template<> struct block_fft_record<124, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    2,   256,   248,  1335>,
        block_fft_implementation<62,    2,   62,    2,   192,   496,  1336>
    >;
};

template<> struct block_fft_record<136, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    4,    64,   272,  1337>,
        block_fft_implementation<34,    2,   34,    4,    60,   544,  1338>
    >;
};

template<> struct block_fft_record<152, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    4,    64,   608,  1339>,
        block_fft_implementation<38,    2,   38,    4,    64,   304,  1340>
    >;
};

template<> struct block_fft_record<160, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,    2,   20,    8,     8,   640,  1341>,
        block_fft_implementation<40,    4,   40,    4,    64,   320,  1342>,
        block_fft_implementation<40,    4,   40,    4,    64,   640,  1343>,
        block_fft_implementation<10,    2,   10,   16,    16,   640,  1344>,
        block_fft_implementation<20,    2,   20,    8,    32,   320,  1345>,
        block_fft_implementation<10,    2,   10,   16,    16,   320,  1346>
    >;
};

template<> struct block_fft_record<168, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    4,    64,   336,  1347>,
        block_fft_implementation<42,    2,   42,    4,    64,   672,  1348>
    >;
};

template<> struct block_fft_record<176, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,    4,   44,    4,    64,   352,  1349>,
        block_fft_implementation<44,    4,   44,    4,    64,   704,  1350>,
        block_fft_implementation<22,    2,   22,    8,     8,   704,  1351>,
        block_fft_implementation<22,    2,   22,    8,    32,   352,  1352>
    >;
};

template<> struct block_fft_record<180, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,    6,   30,    6,    42,   360,  1353>,
        block_fft_implementation<30,    6,   30,    6,    42,   720,  1354>,
        block_fft_implementation<60,    3,   60,    3,    20,   360,  1355>,
        block_fft_implementation<60,    3,   60,    3,    16,   720,  1356>
    >;
};

template<> struct block_fft_record<184, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    4,    64,   736,  1357>,
        block_fft_implementation<46,    2,   46,    4,    64,   368,  1358>
    >;
};

template<> struct block_fft_record<192, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,    8,   24,    8,     8,   384,  1359>,
        block_fft_implementation<24,    8,   24,    8,     8,   768,  1360>,
        block_fft_implementation<12,    4,   12,   16,     4,   768,  1361>,
        block_fft_implementation<12,    4,   12,   16,     4,   384,  1362>,
        block_fft_implementation<48,    4,   48,    4,    64,   384,  1363>,
        block_fft_implementation< 6,    2,    6,   32,     8,   768,  1364>,
        block_fft_implementation<48,    4,   48,    4,    16,   768,  1365>,
        block_fft_implementation< 6,    2,    6,   32,     8,   384,  1366>
    >;
};

template<> struct block_fft_record<200, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   10,   20,   10,     6,   800,  1367>,
        block_fft_implementation<20,   10,   20,   10,     6,   400,  1368>,
        block_fft_implementation<40,    5,   40,    5,    50,   400,  1369>,
        block_fft_implementation<10,    2,   10,   20,    12,   800,  1370>,
        block_fft_implementation<10,    2,   10,   20,    12,   400,  1371>,
        block_fft_implementation<40,    5,   40,    5,    40,   800,  1372>,
        block_fft_implementation<50,    2,   50,    4,    64,   400,  1373>,
        block_fft_implementation<50,    2,   50,    4,    60,   800,  1374>
    >;
};

template<> struct block_fft_record<208, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,    4,   52,    4,    64,   416,  1375>,
        block_fft_implementation<52,    4,   52,    4,    58,   832,  1376>,
        block_fft_implementation<26,    2,   26,    8,    32,   832,  1377>,
        block_fft_implementation<26,    2,   26,    8,    32,   416,  1378>
    >;
};

template<> struct block_fft_record<224, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,    2,   28,    8,    32,   896,  1379>,
        block_fft_implementation<56,    4,   56,    4,    64,   448,  1380>,
        block_fft_implementation<14,    2,   14,   16,     4,   896,  1381>,
        block_fft_implementation<56,    4,   56,    4,    54,   896,  1382>,
        block_fft_implementation<28,    2,   28,    8,     8,   448,  1383>,
        block_fft_implementation<14,    2,   14,   16,    16,   448,  1384>
    >;
};

template<> struct block_fft_record<232, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    4,   128,   464,  1385>,
        block_fft_implementation<58,    2,   58,    4,    96,   928,  1386>
    >;
};

template<> struct block_fft_record<240, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,    4,   60,    4,    64,   480,  1387>,
        block_fft_implementation<60,    4,   60,    4,    50,   960,  1388>,
        block_fft_implementation<30,    2,   30,    8,    32,   960,  1389>,
        block_fft_implementation<30,    2,   30,    8,    32,   480,  1390>
    >;
};

template<> struct block_fft_record<248, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    4,   128,   496,  1391>,
        block_fft_implementation<62,    2,   62,    4,    96,   992,  1392>
    >;
};

template<> struct block_fft_record<252, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    6,   42,    6,    42,   504,  1393>,
        block_fft_implementation<42,    6,   42,    6,    42,  1008,  1394>
    >;
};

template<> struct block_fft_record<272, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,    2,   34,    8,    30,  1088,  1395>,
        block_fft_implementation<34,    2,   34,    8,    32,   544,  1396>
    >;
};

template<> struct block_fft_record<288, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   12,   24,   12,    20,   576,  1397>,
        block_fft_implementation<24,   12,   24,   12,    20,  1152,  1398>,
        block_fft_implementation<12,    2,   12,   24,    10,  1152,  1399>,
        block_fft_implementation<12,    2,   12,   24,    10,   576,  1400>,
        block_fft_implementation<48,    6,   48,    6,    42,   576,  1401>,
        block_fft_implementation<18,    2,   18,   16,    16,  1152,  1402>,
        block_fft_implementation<36,    2,   36,    8,    28,  1152,  1403>,
        block_fft_implementation< 6,    2,    6,   48,     8,  1152,  1404>,
        block_fft_implementation< 6,    2,    6,   48,     8,   576,  1405>,
        block_fft_implementation<36,    2,   36,    8,     8,   576,  1406>,
        block_fft_implementation<48,    6,   48,    6,    42,  1152,  1407>,
        block_fft_implementation<18,    2,   18,   16,     8,   576,  1408>
    >;
};

template<> struct block_fft_record<300, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   10,   30,   10,    24,   600,  1409>,
        block_fft_implementation<30,   10,   30,   10,    24,  1200,  1410>,
        block_fft_implementation<60,    5,   60,    5,    50,   600,  1411>,
        block_fft_implementation<60,    5,   60,    5,    40,  1200,  1412>
    >;
};

template<> struct block_fft_record<304, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,    2,   38,    8,    32,  1216,  1413>,
        block_fft_implementation<38,    2,   38,    8,    32,   608,  1414>
    >;
};

template<> struct block_fft_record<320, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<40,    8,   40,    8,    32,   640,  1415>,
        block_fft_implementation<20,    4,   20,   16,     4,  1280,  1416>,
        block_fft_implementation<20,    4,   20,   16,     4,   640,  1417>,
        block_fft_implementation<40,    8,   40,    8,    32,  1280,  1418>,
        block_fft_implementation<10,    2,   10,   32,     8,  1280,  1419>,
        block_fft_implementation<10,    2,   10,   32,     8,   640,  1420>
    >;
};

template<> struct block_fft_record<336, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,    8,    32,  1344,  1421>,
        block_fft_implementation<42,    2,   42,    8,    32,   672,  1422>
    >;
};

template<> struct block_fft_record<352, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,    2,   44,    8,    32,  1408,  1423>,
        block_fft_implementation<22,    2,   22,   16,    16,  1408,  1424>,
        block_fft_implementation<44,    2,   44,    8,    32,   704,  1425>,
        block_fft_implementation<22,    2,   22,   16,    16,   704,  1426>
    >;
};

template<> struct block_fft_record<360, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,    6,   60,    6,    42,   720,  1427>,
        block_fft_implementation<30,    2,   30,   12,    20,   720,  1428>,
        block_fft_implementation<60,    6,   60,    6,    34,  1440,  1429>,
        block_fft_implementation<30,    2,   30,   12,    20,  1440,  1430>
    >;
};

template<> struct block_fft_record<368, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,    2,   46,    8,    32,   736,  1431>,
        block_fft_implementation<46,    2,   46,    8,    32,  1472,  1432>
    >;
};

template<> struct block_fft_record<384, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,    2,   12,   32,     2,  1536,  1433>,
        block_fft_implementation<12,    2,   12,   32,     2,   768,  1434>,
        block_fft_implementation<24,    2,   24,   16,    16,   768,  1435>,
        block_fft_implementation<48,    8,   48,    8,    32,   768,  1436>,
        block_fft_implementation<24,    2,   24,   16,     4,  1536,  1437>,
        block_fft_implementation<48,    8,   48,    8,    32,  1536,  1438>,
        block_fft_implementation< 6,    2,    6,   64,     4,  1536,  1439>,
        block_fft_implementation< 6,    2,    6,   64,     4,   768,  1440>
    >;
};

template<> struct block_fft_record<392, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   14,   28,   14,    18,   784,  1441>,
        block_fft_implementation<28,   14,   28,   14,    18,  1568,  1442>,
        block_fft_implementation<14,    2,   14,   28,    18,  1568,  1443>,
        block_fft_implementation<56,    7,   56,    7,    36,   784,  1444>,
        block_fft_implementation<14,    2,   14,   28,     2,   784,  1445>,
        block_fft_implementation<56,    7,   56,    7,    30,  1568,  1446>
    >;
};

template<> struct block_fft_record<416, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,    2,   52,    8,    28,  1664,  1447>,
        block_fft_implementation<26,    2,   26,   16,    16,  1664,  1448>,
        block_fft_implementation<52,    2,   52,    8,    32,   832,  1449>,
        block_fft_implementation<26,    2,   26,   16,    16,   832,  1450>
    >;
};

template<> struct block_fft_record<432, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<36,   12,   36,   12,    42,   864,  1451>,
        block_fft_implementation<12,    3,   12,   36,    14,  1728,  1452>,
        block_fft_implementation<36,   12,   36,   12,    18,  1728,  1453>,
        block_fft_implementation<12,    3,   12,   36,    14,   864,  1454>,
        block_fft_implementation<24,    3,   24,   18,    14,   864,  1455>,
        block_fft_implementation<18,    2,   18,   24,    10,  1728,  1456>,
        block_fft_implementation< 6,    2,    6,   72,    10,  1728,  1457>,
        block_fft_implementation<24,    3,   24,   18,    14,  1728,  1458>,
        block_fft_implementation< 6,    2,    6,   72,    10,   864,  1459>,
        block_fft_implementation<18,    2,   18,   24,    10,   864,  1460>,
        block_fft_implementation<48,    3,   48,    9,    28,   864,  1461>,
        block_fft_implementation<54,    2,   54,    8,    28,  1728,  1462>,
        block_fft_implementation<48,    3,   48,    9,    28,  1728,  1463>,
        block_fft_implementation<54,    2,   54,    8,     8,   864,  1464>
    >;
};

template<> struct block_fft_record<448, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,    4,   28,   16,    16,  1792,  1465>,
        block_fft_implementation<28,    4,   28,   16,    16,   896,  1466>,
        block_fft_implementation<56,    8,   56,    8,    32,   896,  1467>,
        block_fft_implementation<56,    8,   56,    8,    26,  1792,  1468>,
        block_fft_implementation<14,    2,   14,   32,     8,  1792,  1469>,
        block_fft_implementation<14,    2,   14,   32,     8,   896,  1470>
    >;
};

template<> struct block_fft_record<464, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,    2,   58,    8,     8,   928,  1471>,
        block_fft_implementation<58,    2,   58,    8,    52,  1856,  1472>
    >;
};

template<> struct block_fft_record<480, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,    2,   30,   16,    16,  1920,  1473>,
        block_fft_implementation<60,    2,   60,    8,    50,  1920,  1474>,
        block_fft_implementation<60,    2,   60,    8,    32,   960,  1475>,
        block_fft_implementation<30,    2,   30,   16,    16,   960,  1476>
    >;
};

template<> struct block_fft_record<496, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,    2,   62,    8,    64,   992,  1477>,
        block_fft_implementation<62,    2,   62,    8,    48,  1984,  1478>
    >;
};

template<> struct block_fft_record<500, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,    5,   10,   50,    10,  2000,  1479>,
        block_fft_implementation<10,    5,   10,   50,    10,  1000,  1480>,
        block_fft_implementation<20,    5,   20,   25,    10,  2000,  1481>,
        block_fft_implementation<20,    5,   20,   25,    10,  1000,  1482>,
        block_fft_implementation<50,   10,   50,   10,    50,  1000,  1483>,
        block_fft_implementation<50,   10,   50,   10,    48,  2000,  1484>
    >;
};

template<> struct block_fft_record<504, half, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,    2,   42,   12,    20,  2016,  1485>,
        block_fft_implementation<42,    2,   42,   12,    20,  1008,  1486>
    >;
};
