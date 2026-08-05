#pragma once

#include <Arduino.h>


struct WordToken {
 const char* word;
 uint8_t id;
};

WordToken vocabulary[] = {
    {"the",2},
    {"you",3},
    {"hey",4},
    {"light",5},
    {"please",6},
    {"could",7},
    {"can",8},
    {"assistant",9},
    {"it",10},
    {"is",11},
    {"lamp",12},
    {"off",13},
    {"what",14},
    {"on",15},
    {"dark",16},
    {"room",17},
    {"too",18},
    {"bright",19},
    {"make",20},
    {"switch",21},
    {"bedroom",22},
    {"time",23},
    {"power",24},
    {"led",25},
    {"play",26},
    {"music",27},
    {"turn",28},
    {"hi",29},
    {"up",30},
    {"tell",31},
    {"me",32},
    {"a",33},
    {"joke",34},
    {"i",35},
    {"weather",36},
    {"kitchen",37},
    {"go",38},
    {"to",39},
    {"sleep",40},
    {"open",41},
    {"door",42},
    {"are",43},
    {"lights",44},
    {"hello",45},
    {"disable",46},
    {"in",47},
    {"here",48},
    {"want",49},
    {"darkness",50},
    {"much",51},
    {"darken",52},
    {"wake",53},
    {"brighten",54},
    {"need",55},
    {"some",56},
    {"its",57},
    {"cannot",58},
    {"see",59},
    {"activate",60},
    {"my",61},
    {"it's",62},
    {"enable",63},
    {"sunny",64},
    {"who",65},
    {"how",66},
    {"increase",67},
    {"temperature",68},
    {"random",69},
    {"command",70},
};



uint8_t lookupWord(String word)
{
    word.toLowerCase();

    for (int i = 0; i < sizeof(vocabulary)/sizeof(vocabulary[0]); i++)
    {
        if (word == vocabulary[i].word)
        {
            return vocabulary[i].id;
        }
    }

    return 1; // <UNK>
}


void tokenize(String text, uint8_t* output, int max_len)
{
    for(int i=0;i<max_len;i++)
    {
        output[i]=0;
    }
    int index=0;
    text.toLowerCase();
    while(text.length() && index < max_len)
    {
        int space = text.indexOf(' ');
        String word;
        if(space == -1)
        {
            word=text;
            text="";
        }
        else
        {
            word=text.substring(0,space);
            text=text.substring(space+1);
        }

        if(word.length())
        {
            output[index++] = lookupWord(word);
        }
    }
}