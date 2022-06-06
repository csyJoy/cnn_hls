#include "conv2.h"


template<int InputChannel, int OutputChannel, int InputSize, int OutputSize, int WeightSize>
void conv2(float *In_ddr, float *W_ddr, float *Out_ddr){
    float In[4][InputSize][InputSize];
#pragma HLS array_partition variable=In complete dim=1
    float Out[4][OutputSize][OutputSize];
#pragma HLS array_partition variable=Out complete dim=1
    float W[4][4][WeightSize][WeightSize];
#pragma HLS array_partition variable=W complete dim=1
#pragma HLS array_partition  variable=W complete dim=2

    Output_Channel_Tiling:
	for(int cho = 0; cho < OutputChannel; cho += 4){
		Input_Channel_Tiling:
		for(int chi = 0; chi < InputChannel; chi += 4){
			// we should load in data first
			for(int L_ri = 0;L_ri < InputSize; L_ri++){
#pragma HLS PIPELINE
				for(int L_ci = 0;L_ri < InputSize; L_ci++){
					for(int L_chi=0; L_chi < 4 && L_chi < InputChannel; L_chi++){
						In[L_chi][L_ri][L_ci] = *In_ddr++;
					}
				}
			}
		    // we should load weight data first
            for(int L_cho = 0; L_cho < 4 && L_cho < OutputChannel; L_cho++){
#pragma HLS PIPELINE
                for(int L_chi = 0; L_chi < 4 && L_chi < InputChannel; L_chi++){
                    for(int L_kr = 0; L_kr < WeightSize; L_kr++){
                        for(int L_kc = 0; L_kc < WeightSize; L_kc++){
                            W[L_cho][L_chi][L_kr][L_kc] = *W_ddr++;
                        }
                    }
                }
            }
            Kernel_Row:
            for(int kr = 0; kr < WeightSize; kr++){
                Kernel_RowColumn:
                for(int kc = 0;kc < WeightSize; kc++){
                    Row:
                    for(int r = 0;r < OutputSize;r++){
                        Column:
                        for(int c = 0;c < OutputSize; c++){
#pragma HLS PIPELINE
                            Output_Channel:
                            for(int cho = 0; cho < CHout && cho < OutputChannel; cho++){
                                Input_Channel:
                                for(int chi = 0; chi < CHin && chi < InputChannel; chi++){
                                    Out[cho][r][c] += In[chi][r + kr][c + kc] * W[cho][chi][kr][kc];
                                }
                            }
                        }
                    }
                }
            }
		}
        for(int L_ro = 0; L_ro < OutputSize; L_ro++){
#pragma HLS PIPELINE
            for(int L_co = 0; L_co < OutputSize; L_co++){
                for(int L_cho = 0; L_cho < CHout && L_cho < OutputChannel;L_cho++){
                    *Out_ddr++ = Out[L_cho][L_ro][L_co];
                }
            }
        }
	}
    return;
}
