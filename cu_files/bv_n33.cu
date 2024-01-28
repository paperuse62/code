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
	sim.append(Simulation::X(32));
	sim.append(Simulation::H(32));
	sim.append(Simulation::CX(1, 32));
	sim.append(Simulation::CX(5, 32));
	sim.append(Simulation::CX(6, 32));
	sim.append(Simulation::CX(7, 32));
	sim.append(Simulation::CX(12, 32));
	sim.append(Simulation::CX(13, 32));
	sim.append(Simulation::CX(14, 32));
	sim.append(Simulation::CX(17, 32));
	sim.append(Simulation::CX(18, 32));
	sim.append(Simulation::CX(19, 32));
	sim.append(Simulation::CX(20, 32));
	sim.append(Simulation::CX(21, 32));
	sim.append(Simulation::CX(22, 32));
	sim.append(Simulation::CX(24, 32));
	sim.append(Simulation::CX(26, 32));
	sim.append(Simulation::CX(27, 32));
	sim.append(Simulation::CX(28, 32));
	sim.append(Simulation::CX(29, 32));
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
	int n_qubits=33;
	int n_gpus=2;
	int chunkSize=25;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
