// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Preamble for the direct-launch kernel translation unit.
//
// The ops-transformer kernel headers include
//   ../mla_prolog_template_tiling_key.h
// which pulls in ascendc/host_api/tiling/template_argument.h.  That header only
// exists to declare the ASCENDC_TPL tiling-key selection (needed on the *host*
// tiling side, which does include it) and it does not compile during the host
// pass of the direct-launch build (it calls printf without including <cstdio>).
//
// The kernel itself only needs the two CV-split constants, so the declare header
// is skipped here via its include guard and the constants are provided locally
// with the same values (MIX_AIC_1_1 = 6, MIX_AIC_1_2 = 7, see the encoder in
// template_argument.h).

#ifndef MLA_PROLOG_V3_KERNEL_PREAMBLE_H
#define MLA_PROLOG_V3_KERNEL_PREAMBLE_H

#ifndef MLA_PROLOG_TEMPLATE_TILING_KEY_H
#define MLA_PROLOG_TEMPLATE_TILING_KEY_H
#endif

#ifndef ASCENDC_TPL_MIX_AIC_1_1
#define ASCENDC_TPL_MIX_AIC_1_1 6
#endif

#ifndef ASCENDC_TPL_MIX_AIC_1_2
#define ASCENDC_TPL_MIX_AIC_1_2 7
#endif

#endif // MLA_PROLOG_V3_KERNEL_PREAMBLE_H
