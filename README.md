# GadgetSVM

A Consensus Algorithm for Linear Support Vector Machines

## Requirements
To run this project, you will need:

* Java JDK 1.8 or later installed on your system
* Eclipse IDE 4.x or later

## Installation
To install and run this project on your local machine, follow these steps:

1. Clone the project repository from GitHub using the following command:                                                                                         
  ```git clone https://github.com/nisargnegi/GadgetSVM.git```
2. Open Eclipse and select "File" > "Import Project" > "Existing Projects into Workspace" and browse to the directory where you cloned the project.
3. Select root directory as project folder & Check the project "GadgetSVM"
4. Select the project and click "Finish".

## Usage 

The Machine Learning project is implemented as a Java application that can be run from Eclipse. The project includes a set of example datasets and SVM models to demonstrate the Consensus based Machine learning for SVM.

To use the project, follow these steps:

1. Open Eclipse and navigate to the "Project" > "src" > "peersim" > "Simulator.java"
2. Right-click on "Simulator.java" and select "Run Configurations"
3. Give any name to the configuration and select the Main class as peersim.Simulator
4. Set the Arguments as the destination location for the configuration file: "example/config-pegasosSido.cfg"
5. The program will execute the chosen algorithm and display the output.
6. To use your own dataset and network configuration for the consensus model, update the "config-pegasosSido.cfg" file in example folder with the correct paths and run the program again.

## Configuration
Config file config-pegasosSido.cfg is present inside the example folder. It can be updated to:
1. Use your own dataset. Currently, "network.node.resourcepath data/sido" is present. "data/sido" contains the training data(t_\*) and test data(tst_\*). 
2. Change the consensus network configuration. 


## Contributing
We welcome contributions to the project. If you would like to contribute, please follow these steps:

1. Fork the project repository.
2. Create a new branch for your changes:
   ```git checkout -b your-branch-name```
4. Make your changes to the project.
5. Commit your changes:
```git commit -m "Your commit message"```
4. Push your changes to your forked repository:
```git push origin your-branch-name```
5. Create a pull request.

We will review your changes and merge them if they meet the project's standards and requirements.




