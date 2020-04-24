classes.py contains basic classes that help organize the VALP. They don't add functionality, but formality.
descriptors.py mainly is a container for the descriptors: VALP and Networks.
evolution.py contains basic operations which are needed for performing evolution on VALPs (this script is not complete). It also contains some functionalities not exclusive of evolution, like data loading.
intelligent_local.py has the necessary tools to perform a local search with the "intelligent" application of the modification operators.
Model_Building.py contains the (tenorflow) implementation of the VALP.
Model_Descriptor.py has the necessary tools to construct valid VALP descriptors (previously defined in descriptors.py)
Networks.py contains the (tensorflow) implementation of DNNs.
small_improvements.py contains the code developed for the OLA paper, including the modification operators.