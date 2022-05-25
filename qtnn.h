#ifndef QTNN_H
#define QTNN_H
// ----------------------------------------------------------------------------|
// Title      : Fast Homomorphic Evaluation of Deep Discretized Neural Networks
// Project    : Demonstrate Fast Fully Homomorphic Evaluation of Encrypted Inputs
//              using Deep Discretized Neural Networks hence preserving Privacy
// ----------------------------------------------------------------------------|
// File       : nn.cpp
// Authors    : Florian Bourse      <Florian.Bourse@ens.fr>
//              Michele Minelli     <Michele.Minelli@ens.fr>
//              Matthias Minihold   <Matthias.Minihold@RUB.de>
//              Pascal Paillier     <Pascal.Paillier@cryptoexperts.com>
//
// Reference  : TFHE: Fast Fully Homomorphic Encryption Library over the Torus
//              https://github.com/tfhe
// ----------------------------------------------------------------------------|
// Description:
//     Showcases how to efficiently evaluate privacy-perserving neural networks.
// ----------------------------------------------------------------------------|
// Revisions  :
// Date        Version  Description
// 2017-11-16  0.3.0    Version for github, referenced by ePrint paper
// ----------------------------------------------------------------------------|


// Includes
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <sys/time.h>
#include "mainwindow.h"
// Multi-processing
#include <sys/wait.h>
#include <unistd.h>
// tfhe-lib
#include "tfhe.h"
#include "tfhe_garbage_collector.h"
#include "tlwe.h"
#include "tgsw.h"
#include "lwesamples.h"
#include "lwekey.h"
#include "lweparams.h"
#include "polynomials.h"


// Defines
#define VERBOSE 1
#define STATISTICS false
#define WRITELATEX false
#define N_PROC 4

// Security constants
#define SECLEVEL 80
#define SECNOISE true
#define SECALPHA pow(2., -20)
#define SEC_PARAMS_STDDEV    pow(2., -30)
#define SEC_PARAMS_n  600                   ///  LweParams
#define SEC_PARAMS_N 1024                   /// TLweParams
#define SEC_PARAMS_k    1                   /// TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -36)   /// TLweParams
#define SEC_PARAMS_BK_BASEBITS 10           /// TGswParams
#define SEC_PARAMS_BK_LENGTH    3           /// TGswParams
#define SEC_PARAMS_KS_STDDEV pow(2., -25)   /// Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  1           /// Key Switching Params
#define SEC_PARAMS_KS_LENGTH   18           /// Key Switching Params

// The expected topology of the provided neural network is 256:30:10
#define NUM_NEURONS_LAYERS 3
#define NUM_NEURONS_INPUT  256
#define NUM_NEURONS_HIDDEN 30
#define NUM_NEURONS_OUTPUT 10

#define CARD_TESTSET 1

// Files are expected in the executable's directory
#define PATH_TO_FILES       "buildotests/test/" //TODO FIXME!
#define FILE_TXT_IMG        "txt_img_test.txt"
#define FILE_TXT_BIASES     "txt_biases.txt"
#define FILE_TXT_WEIGHTS    "txt_weights.txt"
#define FILE_TXT_LABELS     "txt_labels.txt"
#define FILE_LATEX          "results_LaTeX.tex"
#define FILE_STATISTICS     "results_stats.txt"

// Tweak neural network
#define THRESHOLD_WEIGHTS  9
#define THRESHOLD_SCORE -100

#define MSG_SLOTS    700
#define TORUS_SLOTS  400


using namespace std;
TFheGateBootstrappingParameterSet *our_default_gate_bootstrapping_parameters(int minimum_lambda);

int qtnn(int *IMAGE);

void encrypt(int *IMAGE,LweSample * enc_image,struct TFheGateBootstrappingSecretKeySet *secret,const struct LweParams * paras);

void net(struct LweSample * enc_image,struct LweSample * enc_scores,const struct LweBootstrappingKeyFFT *bs_key,const struct LweParams * paras);

int decrypt(struct LweSample *enc_scores,struct TFheGateBootstrappingSecretKeySet *secret);

#endif // NN_H
