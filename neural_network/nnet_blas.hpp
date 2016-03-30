#pragma once

extern "C" void *load_network(const char *filename);
extern "C" int   num_inputs(void *network);
extern "C" int   num_outputs(void *network);
extern "C" int   evaluate_network(void *network, double *input, double *output);
extern "C" int   evaluate_network_multiple(void *network, double *input, int numberInputs, double *output);
extern "C" void  destroy_network(void *network);