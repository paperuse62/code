#include <stdio.h>
#include "bmlqsim.cuh"
//Use the SVSim namespace to enable C++/CUDA APIs
using namespace SVSim;

void prepare_circuit(Simulation &sim)
{
	sim.append(Simulation::H(0));
	sim.append(Simulation::H(1));
	sim.append(Simulation::H(2));
	sim.append(Simulation::H(3));
	sim.append(Simulation::H(4));
	sim.append(Simulation::H(5));
	sim.append(Simulation::H(6));
	sim.append(Simulation::H(7));
	sim.append(Simulation::H(8));
	sim.append(Simulation::H(9));
	sim.append(Simulation::H(10));
	sim.append(Simulation::H(11));
	sim.append(Simulation::H(12));
	sim.append(Simulation::H(13));
	sim.append(Simulation::H(14));
	sim.append(Simulation::H(15));
	sim.append(Simulation::H(16));
	sim.append(Simulation::H(17));
	sim.append(Simulation::H(18));
	sim.append(Simulation::H(19));
	sim.append(Simulation::H(20));
	sim.append(Simulation::H(21));
	sim.append(Simulation::H(22));
	sim.append(Simulation::H(23));
	sim.append(Simulation::H(24));
	sim.append(Simulation::H(25));
	sim.append(Simulation::H(26));
	sim.append(Simulation::H(27));
	sim.append(Simulation::H(28));
	sim.append(Simulation::H(29));
	sim.append(Simulation::H(30));
	sim.append(Simulation::H(31));
	sim.append(Simulation::U1(1.6038498, 0));
	sim.append(Simulation::U1(1.0485427, 1));
	sim.append(Simulation::U1(0.40423, 2));
	sim.append(Simulation::U1(2.5152579, 3));
	sim.append(Simulation::U1(1.621544, 4));
	sim.append(Simulation::U1(0.33195536, 5));
	sim.append(Simulation::U1(1.114746, 6));
	sim.append(Simulation::U1(0.24676399, 7));
	sim.append(Simulation::U1(3.0442893, 8));
	sim.append(Simulation::U1(0.50643241, 9));
	sim.append(Simulation::U1(0.13535734, 10));
	sim.append(Simulation::U1(2.2468305, 11));
	sim.append(Simulation::U1(1.0802584, 12));
	sim.append(Simulation::U1(1.7537201, 13));
	sim.append(Simulation::U1(1.8174775, 14));
	sim.append(Simulation::U1(1.9377797, 15));
	sim.append(Simulation::U1(2.0411707, 16));
	sim.append(Simulation::U1(2.3137966, 17));
	sim.append(Simulation::U1(2.5868047, 18));
	sim.append(Simulation::U1(2.0661731, 19));
	sim.append(Simulation::U1(0.85578987, 20));
	sim.append(Simulation::U1(0.20384853, 21));
	sim.append(Simulation::U1(3.137668, 22));
	sim.append(Simulation::U1(0.50249985, 23));
	sim.append(Simulation::U1(0.6663842, 24));
	sim.append(Simulation::U1(0.13566084, 25));
	sim.append(Simulation::U1(2.8250157, 26));
	sim.append(Simulation::U1(0.17638554, 27));
	sim.append(Simulation::U1(0.78400627, 28));
	sim.append(Simulation::U1(2.7672105, 29));
	sim.append(Simulation::U1(0.29386931, 30));
	sim.append(Simulation::U1(0.5744004, 31));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RZ(0.84565862, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::RZ(0.20670148, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::RZ(2.9813781, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::RZ(1.3192529, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::RZ(1.2898158, 5));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::RZ(0.32191063, 6));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::RZ(2.1993379, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::RZ(0.96462255, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::RZ(0.7183701, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::RZ(0.0039601587, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::RZ(0.90300351, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::RZ(2.2209328, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::RZ(2.927858, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::RZ(1.017836, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::RZ(1.4434924, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::RZ(0.3524348, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::RZ(0.56652442, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::RZ(0.79715733, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::RZ(0.22932746, 19));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::RZ(2.1126302, 20));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::RZ(3.1220866, 21));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::RZ(2.1102662, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::RZ(2.5685652, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::RZ(1.906577, 24));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::RZ(1.9447788, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::CX(25, 26));
	sim.append(Simulation::RZ(3.0248347, 26));
	sim.append(Simulation::CX(25, 26));
	sim.append(Simulation::CX(26, 27));
	sim.append(Simulation::RZ(0.50556934, 27));
	sim.append(Simulation::CX(26, 27));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::RZ(0.18676911, 28));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::RZ(1.4029462, 29));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::CX(29, 30));
	sim.append(Simulation::RZ(0.63976518, 30));
	sim.append(Simulation::CX(29, 30));
	sim.append(Simulation::CX(30, 31));
	sim.append(Simulation::RZ(0.83269926, 31));
	sim.append(Simulation::CX(30, 31));
	sim.append(Simulation::CX(30, 31));
	sim.append(Simulation::RZ(1.556835, 31));
	sim.append(Simulation::CX(30, 31));
	sim.append(Simulation::CX(29, 30));
	sim.append(Simulation::RZ(1.1037512, 30));
	sim.append(Simulation::CX(29, 30));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::RZ(0.2762759, 29));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::RZ(0.56185333, 28));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::CX(26, 27));
	sim.append(Simulation::RZ(1.6192862, 27));
	sim.append(Simulation::CX(26, 27));
	sim.append(Simulation::CX(25, 26));
	sim.append(Simulation::RZ(3.0987718, 26));
	sim.append(Simulation::CX(25, 26));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::RZ(0.39055157, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::RZ(2.5844281, 24));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::RZ(2.1802663, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::RZ(0.58790473, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::RZ(0.42872865, 21));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::RZ(0.89670381, 20));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::RZ(2.8069331, 19));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::RZ(0.30696294, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::RZ(0.86342331, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::RZ(0.16398037, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::RZ(1.9205971, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::RZ(0.14945735, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::RZ(0.17382318, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::RZ(0.54807591, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::RZ(0.14470628, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::RZ(1.6132533, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::RZ(0.3437386, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::RZ(0.60677659, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::RZ(0.89676733, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::RZ(2.5923344, 6));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::RZ(2.570556, 5));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::RZ(1.2507047, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::RZ(1.0499484, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::RZ(0.94790226, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RZ(0.075658089, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RZ(2.2444619, 0));
	sim.append(Simulation::RZ(1.8417527, 1));
	sim.append(Simulation::RZ(2.4635264, 2));
	sim.append(Simulation::RZ(2.6720138, 3));
	sim.append(Simulation::RZ(0.14284556, 4));
	sim.append(Simulation::RZ(0.53004082, 5));
	sim.append(Simulation::RZ(2.5609071, 6));
	sim.append(Simulation::RZ(2.7914051, 7));
	sim.append(Simulation::RZ(0.79014335, 8));
	sim.append(Simulation::RZ(1.8631362, 9));
	sim.append(Simulation::RZ(1.2035721, 10));
	sim.append(Simulation::RZ(0.40878468, 11));
	sim.append(Simulation::RZ(0.96921129, 12));
	sim.append(Simulation::RZ(1.458419, 13));
	sim.append(Simulation::RZ(0.27545952, 14));
	sim.append(Simulation::RZ(0.86230143, 15));
	sim.append(Simulation::RZ(2.4062267, 16));
	sim.append(Simulation::RZ(1.3898969, 17));
	sim.append(Simulation::RZ(2.7556761, 18));
	sim.append(Simulation::RZ(2.5787385, 19));
	sim.append(Simulation::RZ(1.2722026, 20));
	sim.append(Simulation::RZ(2.8788487, 21));
	sim.append(Simulation::RZ(1.2561979, 22));
	sim.append(Simulation::RZ(2.4624398, 23));
	sim.append(Simulation::RZ(0.27176477, 24));
	sim.append(Simulation::RZ(1.2474856, 25));
	sim.append(Simulation::RZ(2.6349191, 26));
	sim.append(Simulation::RZ(3.0916427, 27));
	sim.append(Simulation::RZ(1.2716245, 28));
	sim.append(Simulation::RZ(2.371502, 29));
	sim.append(Simulation::RZ(1.1421818, 30));
	sim.append(Simulation::RZ(2.7555909, 31));
	sim.append(Simulation::H(0));
	sim.append(Simulation::H(1));
	sim.append(Simulation::H(2));
	sim.append(Simulation::H(3));
	sim.append(Simulation::H(4));
	sim.append(Simulation::H(5));
	sim.append(Simulation::H(6));
	sim.append(Simulation::H(7));
	sim.append(Simulation::H(8));
	sim.append(Simulation::H(9));
	sim.append(Simulation::H(10));
	sim.append(Simulation::H(11));
	sim.append(Simulation::H(12));
	sim.append(Simulation::H(13));
	sim.append(Simulation::H(14));
	sim.append(Simulation::H(15));
	sim.append(Simulation::H(16));
	sim.append(Simulation::H(17));
	sim.append(Simulation::H(18));
	sim.append(Simulation::H(19));
	sim.append(Simulation::H(20));
	sim.append(Simulation::H(21));
	sim.append(Simulation::H(22));
	sim.append(Simulation::H(23));
	sim.append(Simulation::H(24));
	sim.append(Simulation::H(25));
	sim.append(Simulation::H(26));
	sim.append(Simulation::H(27));
	sim.append(Simulation::H(28));
	sim.append(Simulation::H(29));
	sim.append(Simulation::H(30));
	sim.append(Simulation::H(31));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=32;
	int n_gpus=2;
	int chunkSize=25;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
