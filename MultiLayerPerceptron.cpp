//
//
//  MultiLayerPerceptron.cpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//


#include "MultiLayerPerceptron.hpp"
#include "Perceptron.hpp"



//....................................................................................................
//Return a new MultiLayerPerceptron object with the specified parameters
MultiLayerPerceptron::MultiLayerPerceptron( std::vector< int > layers, double bias, double eta ):
    layers{layers}, bias{bias}, eta{eta} {
        
        for( int i=0 ; i<layers.size() ; ++i ){
            out_values.emplace_back( std::vector< double >( layers[i],0.0 ) ) ;// layers[i] number of 0s
            network.emplace_back( std::vector< Perceptron >() );
            if( i > 0 )     // the 1st layer represents inputs--which are NOT neurons so skip it
                for( int j=0 ; j<layers.size() ; ++j )
                    network[i].emplace_back( Perceptron( layers[i-1],bias )) ;
        }
}
//....................................................................................................
