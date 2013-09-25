Acoustic Scene Classification
=============================

Description
-----------

This is a submission to the IEEE AASP Challenge: Detection and Classification of Acoustic Scenes and Events.
The scene classification (SC) challenge consists of 10 different scenes of 10 audio files of length
30 seconds each, totaling a number of 100 audio clips. The list
of scenes is: busy street, quiet street, park, open-air market, bus,
subway-train, restaurant, shop/supermarket, office, and subway station.
The goal is to test on a development set that is composed
of audio clips of the same scenes as the training set and determine
what scene the audio clips originated from.

Two algorithms are developed here: The first is based on hidden Markov models (HMMs) and Gaussian mixture models (GMMs).
The features that were used include short time Fourier transform, loudness, and spectral sparsity.
The second algorithm applies a support vector machine (SVM) on a frame-based level.

Complete information about the AASP challenge is available at http://c4dm.eecs.qmul.ac.uk/sceneseventschallenge/

All code here is made freely available under the MIT license.

Setup
-----

- Make sure all the files and folders provided are in MATLAB's current path.
- Set up the Pattern Recognition Toolbox (PRT):
  - Download the PRT: https://github.com/newfolder/PRT
	- Run `prtSetup` in the MATLAB console.
	- If MEX is not set up in your MATLAB installation, first make sure that a supported compiler
	  is present (http://www.mathworks.com/support/compilers/R2012b/) and then run `mex -setup`.
	- Run `prtSetupMex`.
- Sample datasets can be downloaded from the AASP Challenge website: http://c4dm.eecs.qmul.ac.uk/rdr/handle/123456789/29

Usage
-----

- There are two different classification algorithms: `classifyScene` and `classifyScene2`.
- Use by calling `classifyScene('/path/to/trainListFile.txt','/path/to/testListFile.txt','/path/to/output')`
