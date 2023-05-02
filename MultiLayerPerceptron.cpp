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
#include "Activation_Function.hpp"

extern const double bias ;      // bias allows for the decision boundry to be independent of
                                // the origin

//....................................................................................................
//Return a new MultiLayerPerceptron object with the specified parameters
MultiLayerPerceptron::MultiLayerPerceptron( std::vector< int > layers, double eta ): layers{layers}, eta{eta} {
        
    /* Below lets construct the network layer by layer. For each layer we must add a vector of output
    values, and a vector of Perceptrons. */
    for( int i=0 ; i<layers.size() ; ++i ){
        
        output_values.emplace_back( std::vector< double >( layers[i],0.0 ) ) ;
        /* layers[i] is the number of neurons (perceptrons) in the given layer. So here I initialize
        the output values to 0s. E.g. if the 1st hidden layer contains 3 neurons, the vector
        'output_values' corresponding to the 1st layer will contain 3 zeros. */
        
        network.emplace_back( std::vector< Perceptron >() );    // for now just insert an empty vector of neurons
        
        /* Below we add individual Perceptrons (neurons) to each layer of the network. Note that
        the 1st layer represents the input layer--which contain NO neurons so we skip it i>0 */
        if( i > 0 )
            for( int j=0 ; j<layers[ i ] ; ++j )

                 
                /* Our 'network' is a vector of Perceptron vectors--i.e. a network of layers of neurons.
                So below, for each layer we create a Perceptron with as many inputs as there are neurons
                in the previous layer (our network is a fully connected network). Remember, the bias
                input does NOT count here because the Perceptron's constructor already adjusts for it. */
                network[i].emplace_back( layers[i-1] ) ;    /* the beuty of emplace_back allows you to
                call the constructor directly and create the object in-place -- NO unnecessary calls
                to the copy/move constructor (super efficient) -- so we call emplace_back with the same
                arguments as we would normally call the constructor e.g.
                network[i].emplace_back( Perceptron( layers[i-1] )) ; */
    }
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
            output_values[ i ][ j ] = f.sigmoid( (*p).run( output_values[ i-1 ] ) ) ;
            //output_values[ i ][ j ] = f.tanh( (*p).run( output_values[ i-1 ] ) ) ;
        }
    return output_values.back() ;   // return the last layer neutron's output value
}
//....................................................................................................
