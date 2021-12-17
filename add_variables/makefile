CC=g++

CFLAGS=`root-config --cflags` -I/export/nfs0home/kgreif/top_tagging/fj_release/fastjet-install/include
LDLIBS=`root-config --libs` -Wl,-rpath,/export/nfs0home/kgreif/top_tagging/fj_release/fastjet-install/lib -L/export/nfs0home/kgreif/top_tagging/fj_release/fastjet-install/lib -lfastjettools -lfastjet -lm -lEnergyCorrelator -lNsubjettiness
LDFLAGS=`root-config --ldflags`


all: add_variables

add_variables.o: add_variables.cpp
	$(CC) -c add_variables.cpp $(CFLAGS)

Thrust.o: Thrust.cpp
	$(CC) -c Thrust.cpp $(CFLAGS)

BoostToCenterofMass.o: BoostToCenterofMass.cpp
	$(CC) -c BoostToCenterofMass.cpp $(CFLAGS)

add_variables: add_variables.o Thrust.o BoostToCenterofMass.o
	$(CC) -o add_variables add_variables.o Thrust.o BoostToCenterofMass.o $(LDLIBS) $(LDFLAGS)

clean:
	rm add_variables.o
	rm Thrust.o
	rm BoostToCenterofMass.o
	rm ./add_variables

