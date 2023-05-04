//
//
//  MultiLayerPerceptron.cpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//


#include "MultiLayerPerceptron.hpp"
#include "Perceptron.hpp"
#include "Globals.hpp"
#include "Activation_Function.cpp"

extern const double bias ;      // bias allows for the decision boundry to be independent of
                                // the origin

//....................................................................................................
//Return a new MultiLayerPerceptron object with the specified parameters
MultiLayerPerceptron::MultiLayerPerceptron( std::vector< int > shape, double eta ): shape{shape}, eta{eta} {
        
    /* Below lets construct the network layer by layer. For each layer we must add a vector of output
    values, and a vector of Perceptrons. */
    for( int i=0 ; i<shape.size() ; ++i ){          // e.g. shape = {2,2,1}
        
        output_values.emplace_back( std::vector< double >( shape[i],0.0 ) ) ;
        /* shape[i] is the number of neurons (perceptrons) in the given layer. So here I initialize
        the output values to 0s. E.g. if the 1st hidden layer contains 3 neurons, the vector
        'output_values[0]' corresponding to the 1st layer will contain 3 zeros. */
        
        errors.emplace_back( std::vector< double >( shape[i],0.0 ) ) ;
        
        network.emplace_back( std::vector< Perceptron >() );    // for now just insert an empty vector of neurons
        
        /* Below we add individual Perceptrons (neurons) to each layer of the network. Note that
        the 1st layer represents the input layer--which contain NO neurons so we skip it i>0 */
        if( i > 0 )
            for( int j=0 ; j<shape[ i ] ; ++j )

                /* Our 'network' is a vector of Perceptron vectors--i.e. a network of layers of neurons.
                So below, for each layer we create a Perceptron with as many inputs as there are neurons
                in the previous layer (our network is a fully connected network). Remember, the bias
                input does NOT count here because the Perceptron's constructor already adjusts for it. */
                network[i].emplace_back( shape[i-1] ) ;    /* the beuty of emplace_back allows you to
                call the constructor directly and create the object in-place -- NO unnecessary calls
                to the copy/move constructor (super efficient) -- so we call emplace_back with the same
                arguments as we would normally call the Perceptron constructor */
    }
    
    network.shrink_to_fit() ;
    output_values.shrink_to_fit() ;
    errors.shrink_to_fit() ;
}
//....................................................................................................
/* Iterate via the network and print each neuron's weights */
void MultiLayerPerceptron::print_weights() const {

    std::cout << "\n" ;
    for( int i{0} ; i < network.size() ; i++ ){
        if( i==0 ){
            std::cout << "Layer-1 : Input Layer\n" ;
            continue ;
        }
        for( int j{0} ; j < network[ i ].size() ; j++ ){
            
            std::cout << "Layer-"<< i+1 <<" : Neuron-"<< j+1 <<": [ " ;
            
            for( const auto & w: network[ i ][ j ].weights )
                std::cout << w << " " ;
            std::cout << "]\n" ;
        }
    }
    std::cout << std::endl ;
}
//....................................................................................................
/* Given the sequence of weights we initialize the weights for each of the Perceptrons in the network*/
void MultiLayerPerceptron::set_weights( std::vector< std::vector< std::vector< double >>> init_vec ){
    
    for( int i{0} ; i < init_vec.size() ; i++ )
        for( int j{0} ; j < init_vec[i].size() ; j++ )
            
            /* seed each neuron its given set of hardcoded weights -- in a self learning network
            these values can be set at random OR transfered from another network trained on a similar
            task (i.e. Transfer Learning)*/
            network[ i+1 ][ j ].set_weights( init_vec[ i ][ j ] ) ;
}
//....................................................................................................
std::vector<double> MultiLayerPerceptron::run( std::vector< double > inputs ){
    
    output_values[ 0 ] = inputs ;   // given the vector is inputs [ x1,x2 ]
    
    Perceptron * p = nullptr ;
    Activation_Function f ;
    
    for( int i{1} ; i<network.size() ; i++ )
        for( int j{0} ; j<network[ i ].size() ; j++ ){
            
            p = &network[ i ][ j ] ;    // access the given Perceptron
            
            /* run each neuron of the consecutive layer by feeding it the output from each neuron
            of the previous layer */
            output_values[ i ][ j ] = ( (*p).run( output_values[ i-1 ],f.sigmoid ) ) ;
            //output_values[ i ][ j ] = f.tanh( (*p).run( output_values[ i-1 ] ) ) ;
        }
    return output_values.back() ;   // return the last layer neutron's output value(s)
}
//....................................................................................................
double MultiLayerPerceptron::back_prop( std::vector< double > x , std::vector< double > y ) {
    
    /* Step-1:  Given the input feature vector x run the network forward and capture the network’s
                output(s) into vector 'output'. */
    std::vector< double > output = run( x ) ;
    
    /* Step-2:  Using the above network’s 'outputs' and the true label(s) y, calculate the Mean
                Squared Error (MSE). */
    double MSE = 0.0 ;
    
    // The number of output values == number of label values
    for( int i=0 ; i < y.size() ; ++i ){
        MSE += ( (y[ i ]-output[ i ]) * (y[ i ]-output[ i ]) ) ;
    }
    MSE = MSE / shape.back() ;
    
    /* Step-3:  For each Output Neuron, calculate their output δk error terms . This is an
                intermediate error calculation that allows us to gauge how each neuron is doing. */
    
    // Populate the last vector of errors-vector with δs (error terms) of the output layer neurons
    for( int i=0 ; i < output.size() ; ++i ){
        errors.back()[ i ] = output[ i ] * (1 - output[ i ]) * (y[ i ] - output[ i ]) ;
    }
    
    /* Step-4:  For each of the hidden neurons, calculate their δk error terms—we do this backwards
                i.e. we iterate from the last hidden layer, all the way to the first hidden layer
                finding the error term for each hidden neuron. */
    double dot_product = 0.0 ;
    for( int i= int(network.size()-2) ; i > 0 ; --i )        // start from the last hidden layer
                                                            // do NOT confuse with the output layer
        for( int h=0 ; h < network[i].size() ; ++h ){        // iterate via all neurons
            
            dot_product = 0.0 ;
            /* iterate via the weights of the nurons belonging to the layer behind this layer.
            Note that in a fully connected network, a given neuron outputs (the same output value)
            to each of the neurons in the consecutive layer. Ergo, we must iterate as many times
            as there are neurons in the next consecutive layer. Also the weights we must multiply
            by belong to the neurons in the next consecutive layer -- however we only use those
            weights that apply to the outputs coming from the current neuron.*/
            for( int k=0 ; k < shape[ i+1 ] ; ++k )
                
                dot_product += network[i+1][k].weights[h] * errors[i+1][k] ;     // ∑ wkh * δk
            
            /* Complete the formula: error term δh = oh * (1 – oh) * ∑ wkh * δk */
            errors[i][h] = output_values[i][h] * ( 1-output_values[i][h] ) * dot_product ;
        }

    /* Step 5:  Apply the Δ delta rule:
        (1) For each of neuroni’s input, calculate the weight adjustment Δ.
                Δwij = η δi xij

            Δwij = weight adjustment for input_j going into neuron_i
            η = learning rate (for the entire network)
            δi = error term of neuron_i
            xij = value of input_j going into neuron_i
     
        (2) adjust the weights: having all the deltas Δwij , simply add them to ALL the weights wij.
                wij = wij + Δwij

            wij = value of weightj going into neuroni */
    double delta_adjust = 0.0 ;
    for( int i=1 ; i < network.size() ; ++i )           // start from the 1st hidden layer
        for( int j=0 ; j < network[i].size() ; j++ )    // iterate via each neuron (of the given layer)
            
            // iterate via all inputs ; there as many weights as inputs.
            for( int k=0 ; k < network[i][j].input.size() ; ++k ){
                
                if( k < network[i][j].input.size()-1 )
                    delta_adjust = (*this).eta * errors[i][j] * network[i][j].input[ k ]  ;
                else
                    delta_adjust = (*this).eta * errors[i][j] * bias  ;
                
                network[i][j].weights[ k ] += delta_adjust ;
            }
    // Return Mean Squared Error to see if the network needs more training.
    return MSE ;
}
//....................................................................................................
