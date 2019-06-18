#include "arm_compute/graph.h"

#include "support/ToolchainSupport.h"

#include "utils/CommonGraphOptions.h"

#include "utils/GraphUtils.h"

#include "utils/Utils.h"



using namespace arm_compute;

using namespace arm_compute::utils;

using namespace arm_compute::graph::frontend;

using namespace arm_compute::graph_utils;

class GraphMobilefacenetExample : public Example
{
public:
	GraphMobilefacenetExample()
		: cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GraphMobilefacenetExample"), convIndex(1), dconvIndex(1), bnIndex(1)
		//: cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GraphMobilefacenetExample")
	{
	              //  this->convIndex=1;
	               // this->dconvIndex=1;
	               // this->bnIndex=1;
	}

	GraphMobilefacenetExample(const GraphMobilefacenetExample &) = delete;

	GraphMobilefacenetExample &operator=(const GraphMobilefacenetExample &) = delete;

	GraphMobilefacenetExample(GraphMobilefacenetExample &&) = default; // NOLINT

	GraphMobilefacenetExample &operator=(GraphMobilefacenetExample &&) = default;      // NOLINT

	~GraphMobilefacenetExample() override = default;

	bool do_setup(int argc, char **argv) override
	{
		// Parse arguments
		cmd_parser.parse(argc, argv);
		// Consume common parameters
		common_params = consume_common_graph_parameters(common_opts);
		// Return when help menu is requested
		if (common_params.help)
		{
			cmd_parser.print_help(argv[0]);
			return false;
		}
		// Print parameter values
		std::cout << common_params << std::endl;
		
		// Create input descriptor
		const TensorShape tensor_shape = permute_shape(TensorShape(112U, 96U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
		TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);
		// Set graph hints
		graph << common_params.target
			<< DepthwiseConvolutionMethod::Optimized3x3 // TODO(COMPMID-1073): Add heuristics to automatically call the optimized 3x3 method  //Q: meaning?
			<< common_params.fast_math_hint;

		// Create core graph
		if (arm_compute::is_data_type_float(common_params.data_type))
		{
			create_graph_float(input_descriptor);
		}
		else
		{
			create_graph_qasymm8(input_descriptor);   //Q: not used?
		}
		// Create common tail
		graph << ReshapeLayer(TensorShape(128U)).set_name("face_feature")
			//<< OutputLayer(arm_compute::support::cpp14::make_unique<DummyAccessor>(0));
			<< OutputLayer(arm_compute::support::cpp14::make_unique<PrintAccessor>());
		// Finalize graph
		GraphConfig config;
		config.num_threads = common_params.threads;
		config.use_tuner = common_params.enable_tuner;
		//config.tuner_mode = common_params.tuner_mode;   //Q: meaning?
		config.tuner_file = common_params.tuner_file;

		// the problem
		graph.finalize(common_params.target, config);

		return true;

	}

	void do_run() override
	{
		// Run graph
		graph.run();
	}

private:
	CommandLineParser  cmd_parser;
	CommonGraphOptions common_opts;
	CommonGraphParams  common_params;
	Stream             graph;
	int                convIndex;
	int                dconvIndex;
        int                bnIndex;
private:
	enum class IsResidual
	{
		Yes,
		No
	};

	enum class HasExpand
	{
		Yes,
		No
	};
	
private:
	void create_graph_float(TensorDescriptor &input_descriptor)
	{
		// Create model path
		const std::string model_path = "/cnn_data/mobilefacenet_model/";
	        //int  this->convIndex=1;
	        //int  this->dconvIndex=1;
	        //int  this->bnIndex=1;

		// Create a preprocessor object
		//std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

		// Create a preprocessor object
		const std::array<float, 3> mean_rgb{ { 0.0f, 0.0f, 0.0f } };
		const float scale = 1.0/255;
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb, scale,
                                                                                                                   false /* Do not convert to BGR */);
		
		// Get trainable parameters data path
		std::string data_path = common_params.data_path;
		std::string total_path = "cnn_data/";
		std::string param_path = "mobilefacenet_header_";
		if (!data_path.empty())
		{
			data_path += model_path;
		}
		graph << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false))
			<< ConvolutionLayer(3U, 3U, 64U,
			get_weights_accessor(data_path,total_path +"conv"+std::to_string(this->convIndex++)+"_weights.npy", DataLayout::NCHW),
			std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
			PadStrideInfo(2, 2, 1, 0, 1, 0, DimensionRoundingType::CEIL)).set_name("Conv1");
			//PadStrideInfo(2, 2, 1, 1, 1, 1, DimensionRoundingType::FLOOR))
			//.set_name("Conv") << OutputLayer(arm_compute::support::cpp14::make_unique<PrintAccessor>());
		
			graph << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_mean.npy"),
			get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_variance.npy"),
			get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_gamma.npy"),
			get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_beta.npy"),
			0.0010000000474974513f)
			.set_name(param_path + "/BatchNorm_1");
		        this->bnIndex++;
                        //std::cout << "after 1U, 3U" << std::endl;

			graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))  //the scale need to be fixed
			.set_name(param_path + "/regular/PRelu")


		<< DepthwiseConvolutionLayer(3U, 3U,
			get_weights_accessor(data_path, total_path + "dconv"+std::to_string(this->dconvIndex++)+"_weights.npy", DataLayout::NCHW),
			std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
			PadStrideInfo(1, 1, 1, 1, 1, 1,DimensionRoundingType::FLOOR))
			.set_name(param_path + "/depthwise/depthwise")

			<< BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_mean.npy"),
			get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_variance.npy"),
			get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_gamma.npy"),
			get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_beta.npy"),
			0.0010000000474974513f)
			.set_name(param_path + "/BatchNorm_2");
			this->bnIndex++;

			graph<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
			.set_name(param_path + "/depwise/PRelu");
/*
		get_expanded_conv_float(data_path, "expanded_conv", 64U, 64U, PadStrideInfo(2, 2, 0, 0, 0, 0,DimensionRoundingType::FLOOR), 5, 2);
		get_expanded_conv_float(data_path, "expanded_conv_1", 64U, 64U, PadStrideInfo(2, 2, 0, 0,0, 0, DimensionRoundingType::FLOOR), 1, 4);
		get_expanded_conv_float(data_path, "expanded_conv_2", 64U, 128U, PadStrideInfo(1, 1, 0, 0,0 , 0,  DimensionRoundingType::FLOOR), 6, 2);
		get_expanded_conv_float(data_path, "expanded_conv_3", 128U, 128U, PadStrideInfo(2, 2, 0, 0, 0, 0, DimensionRoundingType::FLOOR), 1, 4);
		get_expanded_conv_float(data_path, "expanded_conv_4", 128U, 128U, PadStrideInfo(1, 1, 0, 0, 0, 0, DimensionRoundingType::FLOOR), 2, 2);
*/
			get_expanded_conv_float(data_path, "cnn_data/", 64U, 64U, PadStrideInfo(2, 2, 0, 0, 0, 0,DimensionRoundingType::FLOOR), 5, 2);
		get_expanded_conv_float(data_path, "cnn_data/", 64U, 64U, PadStrideInfo(2, 2, 0, 0,0, 0, DimensionRoundingType::FLOOR), 1, 4);
		get_expanded_conv_float(data_path, "cnn_data/", 64U, 128U, PadStrideInfo(1, 1, 0, 0,0 , 0,  DimensionRoundingType::FLOOR), 6, 2);
		get_expanded_conv_float(data_path, "cnn_data/", 128U, 128U, PadStrideInfo(2, 2, 0, 0, 0, 0, DimensionRoundingType::FLOOR), 1, 4);
		get_expanded_conv_float(data_path, "cnn_data/", 128U, 128U, PadStrideInfo(1, 1, 0, 0, 0, 0, DimensionRoundingType::FLOOR), 2, 2);

		graph << ConvolutionLayer(1U, 1U, 256U,
			get_weights_accessor(data_path, total_path +"conv"+std::to_string(this->convIndex)+"_weights.npy", DataLayout::NCHW),
			std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
			PadStrideInfo(1, 1, 0, 0))
			.set_name("Conv_valid");
			this->convIndex++;

			graph << DepthwiseConvolutionLayer(7U, 6U,
			get_weights_accessor(data_path, total_path +"dconv"+std::to_string(this->dconvIndex++)+"_weights.npy", DataLayout::NCHW),
			std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
			PadStrideInfo(1, 1, 0, 0))
			.set_name(param_path + "/depthwise/depthwise_last")

			<< ConvolutionLayer(1U, 1U, 128U,
			get_weights_accessor(data_path, total_path +"conv"+std::to_string(this->convIndex)+"_weights.npy", DataLayout::NCHW),
			std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
			PadStrideInfo(1, 1, 0, 0))
			.set_name("Conv_valid_last");
		        this->convIndex++;
			
			graph <<FullyConnectedLayer(
			128U,
			get_weights_accessor(data_path,total_path +"fc1_weights.npy", DataLayout::NCHW),
			std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr)).set_name("fc1");
	}

	void get_expanded_conv_float(const std::string &data_path, std::string &&param_path,
			unsigned int input_channels, unsigned int output_channels, PadStrideInfo dwc_pad_stride_info,
			unsigned int expansion_size, int times)
		{
			//std::string total_path = param_path + "_";
			std::string total_path = param_path;
			//SubStream   left(graph);

			// Add expand node
		
			
				graph << ConvolutionLayer(1U, 1U, input_channels * expansion_size,
				get_weights_accessor(data_path, total_path +"conv"+std::to_string(this->convIndex)+"_weights.npy", DataLayout::NCHW),
				std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), dwc_pad_stride_info);
				this->convIndex++;
				graph << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_mean.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_variance.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_gamma.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_beta.npy"),
			    0.0010000000474974513f);
			    this->bnIndex++;
			    graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
			
	
			// Add depthwise node
				<< DepthwiseConvolutionLayer(3U, 3U,
				get_weights_accessor(data_path, total_path +"dconv"+std::to_string(this->dconvIndex++)+"_weights.npy", DataLayout::NCHW),
				std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
				PadStrideInfo(1, 1, 1, 1))
				<< BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_mean.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_variance.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_gamma.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_beta.npy"),
			    0.0010000000474974513f);
			        this->bnIndex++;
			       graph<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

				<< ConvolutionLayer(1U, 1U, output_channels,
				get_weights_accessor(data_path, total_path +"conv"+std::to_string(this->convIndex++)+"_weights.npy", DataLayout::NCHW),
				std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
				.set_name(param_path + "/project/Conv2D")

				<< BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_mean.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_variance.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_gamma.npy"),
			    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_beta.npy"),
			    0.0010000000474974513f);
				this->bnIndex++;
			//SubStream   left(graph);

				for (int i = 0; i < times; i++){
					SubStream   left(graph);
					graph << ConvolutionLayer(1U, 1U, output_channels*expansion_size,
						get_weights_accessor(data_path, total_path +"conv"+std::to_string(this->convIndex++)+"_weights.npy", DataLayout::NCHW),
						std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
						.set_name(param_path + "/project/Conv2D")
						<< BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_mean.npy"),
			            get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_variance.npy"),
			            get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_gamma.npy"),
			            get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_beta.npy"),
			            0.0010000000474974513f);
					this->bnIndex++;

						graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

						<< DepthwiseConvolutionLayer(3U, 3U,
						get_weights_accessor(data_path, total_path +"dconv"+std::to_string(this->dconvIndex++)+"_weights.npy", DataLayout::NCHW),
						std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
						PadStrideInfo(1, 1, 1, 1, 1, 1,DimensionRoundingType::FLOOR))
						<< BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_mean.npy"),
			    		get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_variance.npy"),
			    		get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_gamma.npy"),
			    		get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_beta.npy"),
			    		0.0010000000474974513f);
						this->bnIndex++;
						graph<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

						<< ConvolutionLayer(1U, 1U, output_channels,
						get_weights_accessor(data_path, total_path +"conv"+std::to_string(this->convIndex++)+"_weights.npy", DataLayout::NCHW),
						std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
						.set_name(param_path + "/project/Conv2D")

						<< BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_mean.npy"),
					    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_moving_variance.npy"),
					    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_gamma.npy"),
					    get_weights_accessor(data_path, total_path +  "BatchNorm_"+std::to_string(this->bnIndex)+"_beta.npy"),
					    0.0010000000474974513f);
					    this->bnIndex++;

						// Add residual node
						SubStream right(graph);
					    graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(param_path + "/add");
					   // left = graph;
				}
		
		}

	void create_graph_qasymm8(TensorDescriptor &input_descriptor)
        {
	}

};
int main(int argc, char **argv)
{
        int i = 0;
	std::cout << "g" + std::to_string(i++) + "g" << std::endl;
	return arm_compute::utils::run_example<GraphMobilefacenetExample>(argc, argv);
}
