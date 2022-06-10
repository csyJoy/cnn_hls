#include "mnist.h"
#include <cstring>

float expf(float x) {
 x = 1.0 + x / 1024;
 x *= x; x *= x; x *= x; x *= x;
 x *= x; x *= x; x *= x; x *= x;
 x *= x; x *= x;
 return x;
}

float relu(float x){
    return  x > 0 ? x : 0;
}
 
template<int N>
void reluN(float *x){
    for(int i = 0;i < N;i++){
        x[i] = relu(x[i]);
    }
}

template<int I, int O>
void matrixMulti(float *in, float *w, float *o){
    for(int i = 0;i < O;i++){
#pragma HLS UNROLL
        for(int j = 0;j < I;j++){
            o[i] += w[i * I + j] * in[j];
        }
    }
}
float maxPooling(float in[4]){
    float res = -9999;
    for(int i = 0;i < 4;i++){
        if(in[i] > res){
            res = in[i];
        } 
    }
    return res;
}
template<int I>
void maxPoolingN(float *in, float *out){
    float tmp[4];
    for(int i = 0;i < I / 2;i++){
        for(int j = 0;j < I / 2;j++){
            tmp[0] = in[2 * i * I + j];
            tmp[1] = in[2 * i * I + j + 1];
            tmp[2] = in[2 * (i + 1) * I + j];
            tmp[3] = in[2 * (i + 1) * I + j + 1];
            out[i * (I / 2) + j] = maxPooling(tmp);
        }
    }
}
int Softmax_1_8(float input[10],float *probability,float *res){
	int index;
	float sum = 0;
	for(index = 0; index < 10; index++ ){
		probability[index] = expf(input[index]/1000);
		sum += probability[index];
	}
	int max_index = 0;
	for(index = 0; index < 10; index++ ){
			res[index] = probability[index]/sum;
			float res1 = res[index];
			float res2 = res[max_index];
			if(res1 > res2){
				max_index = index;
			}
	}
	return max_index;
}

void mnist(float *in_ddr, int *r){
#pragma HLS INTERFACE m_axi depth=784 port=in_ddr offset=slave bundle=MASTER_BUS
#pragma HLS INTERFACE s_axilite port=r bundle=CRTL_BUS
    float image[784];
    memcpy(image, (const float*)in_ddr, 784 * sizeof(float));
    float out_1[16 * 24* 24];
    conv2<1, 16, 28, 24, 5>(image, conv_w1, out_1);
    reluN<9216>(out_1);
    float out_2[16 * 12 * 12];
    maxPoolingN<24>(out_1, out_2);
    float out_3[32 * 8 * 8];
    conv2<16, 32, 12, 8, 5>(out_2, conv_w2, out_3);
    reluN<2048>(out_3);
    float out_4[32 * 4 * 4];
    maxPoolingN<8>(out_3, out_4);
    float out_5[128];
    matrixMulti<512, 128>(out_4, fc_w1, out_5);
    float out_6[10];
    float prob[10];
    float res[10];
    matrixMulti<128, 10>(out_5, fc_w2, out_6);
    *r = Softmax_1_8(out_6, prob, res);
}


