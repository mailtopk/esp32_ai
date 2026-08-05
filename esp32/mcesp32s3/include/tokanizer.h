#pragma once

#include <Arduino.h>


struct WordToken {
    const char* word;
    uint8_t id;
};

static const WordToken vocabulary[] = {
    {'the': 2}, {'you': 3}, 
    {'hey': 4}, {'light': 5}, 
    {'please': 6}, {'could': 7}, {'can': 8}, {'assistant': 9}, 
    {'it': 10}, {'is': 11}, {'lamp': 12}, {'off': 13},
    {'what': 14}, {'on': 15}, {'dark': 16}, {'room': 17}, 
    {'too': 18}, {'bright': 19}, {'make': 20}, {'switch': 21}, 
    {'bedroom': 22}, {'time': 23}, {'power': 24}, {'led': 25},
    {'play': 26}, {'music': 27}, {'turn': 28}, {'hi': 29}, {'up': 30}, 
    {'tell': 31}, {'me': 32}, {'a': 33}, {'joke': 34}, {'i': 35},
    {'weather': 36}, {'kitchen': 37}, {'go': 38}, {'to': 39}, 
    {'sleep': 40}, {'open': 41}, {'door': 42}, {'are': 43}, 
    {'lights': 44}, {'hello': 45}, {'disable': 46}, {'in': 47}, 
    {'here': 48}, {'want': 49}, {'darkness': 50}, {'good': 51},
    {'morning': 52}, {'night': 53}, {'bed': 54}, {'living': 55}, 
    {'room.': 56}, {'livingroom': 57}, {'turning': 58}, {'off.': 59},
    {'on.': 60}, {'off.': 61}, {'on.': 62}, {'off.': 63}
}


const int VOCAB_SIZE = sizeof(vocabulary) / sizeof(vocabulary[0]);
uint8_t lookupWord(String word)
{
    word.toLowerCase();
    for(int i=0;i<VOCAB_SIZE;i++)
    {
        if(word == vocabulary[i].word)
        {
            return vocabulary[i].id;
        }
    }


    // unknown word
    return 1;
}