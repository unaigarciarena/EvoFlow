# EvoFlow

**This tool provides the necessary facilities for the evolution of DNN structures with different needs, objectives, and complexities.**

## Main library dependencies

This tool has the basic dependencies of numpy and functiools. However, the largest part of the library and core lay on tensorflow for the model creation and evaluation, and on DEAP for the evolutionary component.

## Library characteristics

It is highly customizable, and its usage does not require great familiarity with any of the libraries above.

Right now, EvoFlow *just* performs structural evolution, and weights are not included within the process, even though several ways
of incorporating this feature can be included.

To start understanding the way EvoFlow works, take a look at the .pdf diagram.

The implementation has a main NetowrkDescriptor object from which every evolveble object inherits. These descriptors are based
on Python lists, and contain the structural information of the DNNs. For evaluating these descriptors, we implement another 
main Network class. Each Descriptor child class has a Network counterpart, e.g., MLPDescriptor has MLP, CNNDescriptor has CNN, and so forth.
The Network classes *transform* the structures into actual tensorflow models which can be trained an evaluated. This evaluation
is used as fitness for the structure.

Along with the implementation and the diagram, we provide a set of examples:

1. The simplest one, simple.py, evolves a single MLP for a simple classification problem
2. Adding a little bit of complexity, sequential.py places two MLPs in sequential form for the same classification problem
3. In the next step, we change the first MLP for a CNN, in what could be a simple traditional CNN classification model
4. GAN.py presents a more complicated way of MLP interaction, as the complexity of the  model creation and the loss function definition considerably increases
5. CNNAE.py finally defines a convolutional autoencoder, where trasposed convolutional DNNs are applied.
