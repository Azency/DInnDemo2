#include "qtnn.h"

using namespace std;



void deleteTensor(int*** tensor, int dim_mat, const int* dim_vec);
void deleteMatrix(int**  matrix, int dim_mat);


/// Generate gate bootstrapping parameters for FHE_NN
TFheGateBootstrappingParameterSet *our_default_gate_bootstrapping_parameters(int minimum_lambda)
{
    if (minimum_lambda > 128)
        cerr << "Sorry, for now, the parameters are only implemented for about 128bit of security!\n";

    static const int n = SEC_PARAMS_n;
    static const int N = SEC_PARAMS_N;
    static const int k = SEC_PARAMS_k;
    static const double max_stdev = SEC_PARAMS_STDDEV;

    static const int bk_Bgbit    = SEC_PARAMS_BK_BASEBITS;  //<-- ld, thus: 2^10
    static const int bk_l        = SEC_PARAMS_BK_LENGTH;
    static const double bk_stdev = SEC_PARAMS_BK_STDDEV;

    static const int ks_basebit  = SEC_PARAMS_KS_BASEBITS; //<-- ld, thus: 2^1
    static const int ks_length   = SEC_PARAMS_KS_LENGTH;
    static const double ks_stdev = SEC_PARAMS_KS_STDDEV;


    LweParams  *params_in    = new_LweParams (n,    ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}

void encrypt(int *IMAGE,struct LweSample * enc_image,struct TFheGateBootstrappingSecretKeySet *secret,const struct LweParams * paras)
{

       // Security
    const int minimum_lambda = SECLEVEL;
    const bool noisyLWE      = SECNOISE;
    const double alpha       = SECALPHA;

    // Input data
    const int n_images = CARD_TESTSET;

    // Network specific
    const int num_wire_layers = NUM_NEURONS_LAYERS - 1;
    const int num_neuron_layers = NUM_NEURONS_LAYERS;
    const int num_neurons_in = NUM_NEURONS_INPUT;
    const int num_neurons_hidden = NUM_NEURONS_HIDDEN;
    const int num_neurons_out = NUM_NEURONS_OUTPUT;

    // Vector of number of neurons in layer_in, layer_H1, layer_H2, ..., layer_Hd, layer_out;
    const int topology[num_neuron_layers] = {num_neurons_in, num_neurons_hidden, num_neurons_out};

    const int space_msg = MSG_SLOTS;
    const int space_after_bs = TORUS_SLOTS;

    const bool clamp_biases  = false;
    const bool clamp_weights = false;

    const bool statistics        = STATISTICS;
    const bool writeLaTeX_result = WRITELATEX;

    const int threshold_biases  = THRESHOLD_WEIGHTS;
    const int threshold_weights = THRESHOLD_WEIGHTS;
    const int threshold_scores  = THRESHOLD_SCORE;

    // Program the wheel to value(s) after Bootstrapping
    const Torus32 mu_boot = modSwitchToTorus32(1, space_after_bs);

    const int total_num_hidden_neurons = n_images * NUM_NEURONS_HIDDEN;  //TODO (sum all num_neurons_hidden)*n_images
    const double avg_bs  = 1./NUM_NEURONS_HIDDEN;
    const double avg_total_bs  = 1./total_num_hidden_neurons;
    const double avg_img = 1./n_images;
    const double clocks2seconds = 1. / CLOCKS_PER_SEC;
    const int slice = 1;

    // Huge arrays
    int ** images  = new int* [n_images];
    int  * labels  = new int  [n_images];

    // Temporary variables
    string line;
    int el, l;
    int num_neurons_current_layer_in, num_neurons_current_layer_out;


    if (VERBOSE)
    {
        cout << "Starting experiment to classify " << n_images;
        if (!noisyLWE) cout << " noiseless";
        cout << " encrypted MNIST images." << endl;
//        cout << "(Run: " << argv[0] << " )" << endl;
        cout << "Execution with parameters... alpha = " << alpha << ", number of processes: " << N_PROC << endl;
    }

    if (VERBOSE) cout << "Generate parameters for a security level (according to  CGGI16a) of at least " << minimum_lambda << " [bit]." << endl;

    const LweParams *in_out_params   = paras;

    if (VERBOSE) cout << "Generate the secret keyset." << endl;
    const LweBootstrappingKeyFFT *bs_key = secret->cloud.bkFFT;



    //读取黑白16*16的图片，这一块可以改造，把image[0]=我输入的图片就好了256位
    images[0]=IMAGE;
    cout<<"Complete Reading"<<endl;
    labels[0]=5;


     if (VERBOSE) cout << "Import done. END OF IMPORT" << endl;



    // Temporary variables and Pointers to existing arrays for convenience
    bool notSameSign;
    Torus32 mu, phase;

    int** weight_layer;
    int * bias;
    int * image;
    int pixel, label;
    int x, w, w0;

//*enc_image,这个就是我们要加密的图片，完了直接赋给成员enc_m_image




    for (int img = 0; img < 1; /*img*/ )
    {
        image = images[img];
        label = labels[img++];

        // Generate encrypted inputs for NN (LWE samples for each image's pixels on the fly)
        // To be generic...
        num_neurons_current_layer_out= topology[0];
        num_neurons_current_layer_in = num_neurons_current_layer_out;

        for (int i = 0; i < num_neurons_current_layer_in; ++i)
        {
            pixel = image[i];
            mu = modSwitchToTorus32(pixel, space_msg);
            if (noisyLWE)
            {
                lweSymEncrypt(enc_image + i, mu, alpha, secret->lwe_key);
            }
            else
            {
                lweNoiselessTrivial(enc_image + i, mu, in_out_params);
            }
        }

    }

    
//    for(int i=0;i<256;i++)
//    {
//        ENC_IMAGE[i].a = enc_image[i].a;
//        ENC_IMAGE[i].b = enc_image[i].b;
//        ENC_IMAGE[i].current_variance = enc_image[i].current_variance;
//    }

}

void net(struct LweSample * enc_image,struct LweSample * enc_scores,const LweBootstrappingKeyFFT *bs_key,const struct LweParams * paras)
{
// Security
    const int minimum_lambda = SECLEVEL;
    const bool noisyLWE      = SECNOISE;
    const double alpha       = SECALPHA;

    // Input data
    const int n_images = CARD_TESTSET;
    
    // Network specific
    const int num_wire_layers = NUM_NEURONS_LAYERS - 1;
    const int num_neuron_layers = NUM_NEURONS_LAYERS;
    const int num_neurons_in = NUM_NEURONS_INPUT;
    const int num_neurons_hidden = NUM_NEURONS_HIDDEN;
    const int num_neurons_out = NUM_NEURONS_OUTPUT;

    // Vector of number of neurons in layer_in, layer_H1, layer_H2, ..., layer_Hd, layer_out;
    const int topology[num_neuron_layers] = {num_neurons_in, num_neurons_hidden, num_neurons_out};

    const int space_msg = MSG_SLOTS;
    const int space_after_bs = TORUS_SLOTS;

    const bool clamp_biases  = false;
    const bool clamp_weights = false;

    const bool statistics        = STATISTICS;
    const bool writeLaTeX_result = WRITELATEX;

    const int threshold_biases  = THRESHOLD_WEIGHTS;
    const int threshold_weights = THRESHOLD_WEIGHTS;
    const int threshold_scores  = THRESHOLD_SCORE;

    // Program the wheel to value(s) after Bootstrapping
    const Torus32 mu_boot = modSwitchToTorus32(1, space_after_bs);

    const int total_num_hidden_neurons = n_images * NUM_NEURONS_HIDDEN;  //TODO (sum all num_neurons_hidden)*n_images
    const double avg_bs  = 1./NUM_NEURONS_HIDDEN;
    const double avg_total_bs  = 1./total_num_hidden_neurons;
    const double avg_img = 1./n_images;
    const double clocks2seconds = 1. / CLOCKS_PER_SEC;
    const int slice = 1;

    // Huge arrays
    int*** weights = new int**[num_wire_layers];  // allocate and fill matrices holding the weights
    int ** biases  = new int* [num_wire_layers];  // allocate and fill vectors holding the biases
    int  * labels  = new int  [n_images];

    // Temporary variables
    string line;
    int el, l;
    int num_neurons_current_layer_in, num_neurons_current_layer_out;


    if (VERBOSE)
    {
        cout << "Starting experiment to classify " << n_images;
        if (!noisyLWE) cout << " noiseless";
        cout << " encrypted MNIST images." << endl;
//        cout << "(Run: " << argv[0] << " )" << endl;
        cout << "Execution with parameters... alpha = " << alpha << ", number of processes: " << N_PROC << endl;
    }
    const LweParams *in_out_params  = paras;



    if (VERBOSE) cout << "IMPORT PIXELS, WEIGHTS, BIASES, and LABELS FROM FILES" << endl;



    if (VERBOSE) cout << "Reading weights from " << FILE_TXT_WEIGHTS << endl;
    ifstream file_weights(FILE_TXT_WEIGHTS);
    //读取网络的明文参数

    num_neurons_current_layer_out = topology[0];
    for (l=0; l<num_wire_layers; ++l)
    {
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out = topology[l+1];

        weights[l] = new int*[num_neurons_current_layer_in];
        for (int i = 0; i<num_neurons_current_layer_in; ++i)
        {
            weights[l][i] = new int[num_neurons_current_layer_out];
            for (int j=0; j<num_neurons_current_layer_out; ++j)
            {
                getline(file_weights, line);
                el = stoi(line);
                if (clamp_weights)
                {
                    if (el < -threshold_weights)
                        el = -threshold_weights;
                    else if (el > threshold_weights)
                        el = threshold_weights;
                    // else, nothing as it holds that: -threshold_weights < el < threshold_weights
                }
                weights[l][i][j] = el;
            }
        }
    }
    file_weights.close();


    if (VERBOSE) cout << "Reading biases from " << FILE_TXT_BIASES << endl;
    ifstream file_biases(FILE_TXT_BIASES);

    num_neurons_current_layer_out = topology[0];
    for (l=0; l<num_wire_layers; ++l)
    {
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out = topology[l+1];

        biases [l] = new int [num_neurons_current_layer_out];
        for (int j=0; j<num_neurons_current_layer_out; ++j)
        {
            getline(file_biases, line);
            el = stoi(line);
            if (clamp_biases)
            {
                if (el < -threshold_biases)
                    el = -threshold_biases;
                else if (el > threshold_biases)
                    el = threshold_biases;
                // else, nothing as it holds that: -threshold_biases < el < threshold_biases
            }
            biases[l][j] = el;
        }
    }
    file_biases.close();

    //这一步是读取label，我们用不到，可以直接赋给一个值
    labels[0]=5;


     if (VERBOSE) cout << "Import done. END OF IMPORT" << endl;



    // Temporary variables and Pointers to existing arrays for convenience
    bool notSameSign;
    Torus32 mu, phase;

    int** weight_layer;
    int * bias;
    int * image;
    int pixel, label;
    int x, w, w0;

    LweSample *multi_sum, *bootstrapped ;


    int multi_sum_clear[num_neurons_hidden];
    int output_clear   [num_neurons_out];

    int max_score = 0;
    int max_score_clear = 0;
    int class_enc = 0;
    int class_clear = 0;
    int score = 0;
    int score_clear = 0;


    bool failed_bs = false;
    // Counters
    int count_errors = 0;
    int count_errors_with_failed_bs = 0;
    int count_disagreements = 0;
    int count_disagreements_with_failed_bs = 0;
    int count_disag_pro_clear = 0;
    int count_disag_pro_hom = 0;
    int count_wrong_bs = 0;

    int r_count_errors, r_count_disagreements, r_count_disag_pro_clear, r_count_disag_pro_hom, r_count_wrong_bs, r_count_errors_with_failed_bs, r_count_disagreements_with_failed_bs;
    double r_total_time_network, r_total_time_bootstrappings;

    // For statistics output
    double avg_time_per_classification = 0.0;
    double avg_time_per_bootstrapping = 0.0;
    double total_time_bootstrappings = 0.0;
    double total_time_network = 0.0;
    double error_rel_percent = 0.0;

    // Timings
    clock_t bs_begin, bs_end, net_begin, net_end;
    double time_per_classification, time_per_bootstrapping, time_bootstrappings;



        // ========  FIRST LAYER(S)  ========
        net_begin = clock();

        num_neurons_current_layer_out=topology[0];
        multi_sum = new_LweSample_array(30, in_out_params);
        for (l=0; l<num_wire_layers - 1 ; ++l)     // Note: num_wire_layers - 1 iterations; last one is special. Access weights from level l to l+1.
        {
            // To be generic...
            num_neurons_current_layer_in = num_neurons_current_layer_out;
            num_neurons_current_layer_out= topology[l+1];
            bias = biases[l];
            weight_layer = weights[l];
            for (int j=0; j<num_neurons_current_layer_out; ++j)
            {
                w0 = bias[j];
                multi_sum_clear[j] = w0;

                mu = modSwitchToTorus32(w0, space_msg);
                lweNoiselessTrivial(multi_sum + j, mu, in_out_params);  // bias in the clear

                for (int i=0; i<num_neurons_current_layer_in; ++i)
                {
                    x = image [i];  // clear input
                    w = weight_layer[i][j];  // w^dagger
                    multi_sum_clear[j] += x * w; // process clear input
                    lweAddMulTo(multi_sum + j, w, enc_image + i, in_out_params); // process encrypted input
                }
            }
        }

        // Bootstrap multi_sum
        bootstrapped = new_LweSample_array(num_neurons_current_layer_out, in_out_params);
        bs_begin = clock();
        for (int j=0; j<num_neurons_current_layer_out; ++j)
        {
            /**
             *  Bootstrapping results in each coordinate 'bootstrapped[j]' to contain an LweSample
             *  of low-noise (= fresh LweEncryption) of 'mu_boot*phase(multi_sum[j])' (= per output neuron).
             */
            tfhe_bootstrap_FFT(bootstrapped + j, bs_key, mu_boot, multi_sum + j);
        }
        bs_end = clock();
        time_bootstrappings = bs_end - bs_begin;
        total_time_bootstrappings += time_bootstrappings;
        time_per_bootstrapping = time_bootstrappings*avg_bs;
        if (VERBOSE) cout <<  time_per_bootstrapping*clocks2seconds << " [sec/bootstrapping]" << endl;

        delete_LweSample_array(num_neurons_current_layer_out, multi_sum);  // TODO delete or overwrite after use?

        failed_bs = false;


        // ========  LAST (SECOND) LAYER  ========
        max_score = threshold_scores;
        max_score_clear = threshold_scores;

        bias = biases[l];
        weight_layer = weights[l];
        l++;
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out= topology[l]; // l == L = 2
        //multi_sum = new_LweSample_array(num_neurons_current_layer_out, in_out_params); // TODO possibly overwrite storage
        for (int j=0; j<10; ++j)//num_neurons_current_layer_out
        {
            w0 = bias[j];
            output_clear[j] = w0;
            mu = modSwitchToTorus32(w0, space_after_bs);

            lweNoiselessTrivial(enc_scores + j, mu, in_out_params);

            for (int i=0; i<num_neurons_current_layer_in; ++i)
            {
                w = weight_layer[i][j];
                lweAddMulTo(enc_scores + j, w, bootstrapped + i, in_out_params); // process encrypted input
                // process clear input
                if (multi_sum_clear[i] < 0)
                    output_clear[j] -= w;
                else
                    output_clear[j] += w;

            }
        }

//        for (int j=0;j<10;j++)
//        {
//            enc_scores[j].a=multi_sum[j].a;
//            enc_scores[j].b=multi_sum[j].b;
//            enc_scores[j].current_variance=multi_sum[j].current_variance;
//        }


        net_end = clock();
        time_per_classification = net_end - net_begin;
        total_time_network += time_per_classification;
        if (VERBOSE) cout << "            "<< time_per_classification*clocks2seconds <<" [sec/classification]" << endl;
        // free memory
//        delete_LweSample_array(num_neurons_in,     enc_image);
//        delete_LweSample_array(num_neurons_hidden, bootstrapped);
//        delete_LweSample_array(num_neurons_out,    multi_sum);

}

int decrypt(struct LweSample * enc_scores,struct TFheGateBootstrappingSecretKeySet *secret)
{
    int max_score = -1000;
    int score = 0;
    int class_enc=0;
    for (int j=0; j<10; ++j)//num_neurons_current_layer_out
    {
        score = lwePhase(enc_scores + j, secret->lwe_key);
        if (score > max_score)
        {
            max_score = score;
            class_enc = j;
            //给加密图片分类的结果，在这儿！
        }


    }
    return class_enc;
}

void deleteTensor(int*** tensor, int dim_tensor, const int* dim_vec)
{
    int** matrix;
    int dim_mat;
    for (int i=0; i<dim_tensor; ++i)
    {
        matrix =  tensor[i];
        dim_mat = dim_vec[i];
        deleteMatrix(matrix, dim_mat);
    }
    delete[] tensor;
}


void deleteMatrix(int** matrix, int dim_mat)
{
    for (int i=0; i<dim_mat; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}







