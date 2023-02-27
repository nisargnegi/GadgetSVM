/*
 * Peersim-Gadget : A Gadget protocol implementation in peersim based on the paper
 * Chase Henzel, Haimonti Dutta
 * GADGET SVM: A Gossip-bAseD sub-GradiEnT SVM Solver   
 * 
 * Copyright (C) 2012
 * Deepak Nayak 
 * Columbia University, Computer Science MS'13
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */

package peersim.gossip;

import peersim.core.*;
import peersim.config.*;
/**
 * Class FinalControl
 * Implements {@link Control} interface. This control is run at the end of simulation.
 * Currently it handles the writing of global weight vector to file. But it can be
 * extended to do any post-simulation cleanup job.
 * @author Deepak Nayak
 *
 */
public class FinalControl  implements Control {

//--------------------------------------------------------------------------
// Constants
//--------------------------------------------------------------------------

/** 
 * String name of the parameter used to select the protocol to operate on
 */
public static final String PAR_PROTID = "protocol";


//--------------------------------------------------------------------------
// Fields
//--------------------------------------------------------------------------

/** The name of this object in the configuration file */
private final String name;

/** Protocol identifier */
private final int pid;

// iterator counter
private static int i = 0;


//--------------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------------

/**
 * Creates a new observer and initializes the configuration parameter.
 */
public FinalControl(String name) {
	this.name = name;
	this.pid = Configuration.getPid(name + "." + PAR_PROTID);
  }


//--------------------------------------------------------------------------
// Methods
//--------------------------------------------------------------------------

// Comment inherited from interface
// Do nothing, just for test
public boolean execute() {
	System.out.println("Running final control");

//	final int len = Network.size();
//	for (int i = 0; i <  len; i++) 
//	{
//		LibLinearNode node = (LibLinearNode) Network.get(i);
//		node.writeGlobalWeights();
//		System.out.println("Global weights have been written to "+node.getResourcePath()+"\\global_" + i + ".dat");
//	}
//	
	return false;
}

//--------------------------------------------------------------------------

}

