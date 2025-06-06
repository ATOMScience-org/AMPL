##########################
ATOM Tutorial Introduction
##########################

*Published: June, 2024, ATOM DDM Team*

------------

Welcome to the `ATOM Modeling PipeLine (AMPL) <https://github.com/ATOMScience-org/AMPL>`_ tutorial series. Our tutorial series is set up for our user 
community to take a hands-on approach to employing `AMPL <https://github.com/ATOMScience-org/AMPL>`_ in a step-by-step guide. These tutorials assume 
that you are an intermediate Python user or new to machine learning to build a foundational framework that 
you can use to do meaningful work.
 
The tutorials present an end-to-end pipeline that builds machine learning models for predicting chemical 
properties. We have created easy to follow tutorials that walk through the steps necessary to install 
`AMPL <https://github.com/ATOMScience-org/AMPL>`_, curate a dataset, effectively train and evaluate a machine 
learning model, and use that model to make predictions.

In addition to our written tutorials, we now provide a series of video tutorials on our YouTube channel, `ATOMScience-org <https://www.youtube.com/channel/UCOF6zZ7ltGwopYCoOGIFM-w>`_.  These videos are created to assist users in exploring and leveraging AMPL's robust capabilities.

End-to-End Modeling Pipeline Tutorial Series
********************************************

* Tutorial 1: Data Curation
* Tutorial 2: Splitting Datasets for Validation and Testing
* Tutorial 3: Train a Simple Regression Model
* Tutorial 4: Application of a Trained Model
* Tutorial 5: Hyperparameter Optimization 
* Tutorial 6: Comparing models to select the best hyperparameters
* Tutorial 7: Train a Production Model
* Tutorial 8: Visualizations of Model Performance

Tutorial Series on YouTube
**************************

.. image:: ../_static/img/ampl_intro_video.png
   :target: https://www.youtube.com/watch?v=GIjT7tP0CBw
   :alt: Intro to AMPL YouTube Video
   :width: 160px
   :height: 100px

.. image:: ../_static/img/tutorial_1_video.png
   :target: https://www.youtube.com/watch?v=a-uRfjF8izs
   :alt: Tutorial 1: Data Curation YouTube Video
   :width: 160px
   :height: 100px

.. image:: ../_static/img/tutorial_2_video.png
   :target: https://www.youtube.com/watch?v=gsa2xfG3OSE
   :alt: Tutorial 2: Splitting Datasets for Validation and Testing
   :width: 160px
   :height: 100px

.. image:: ../_static/img/tutorial_3_video.png
   :target: https://www.youtube.com/watch?v=46PhwXqqnyg
   :alt: Tutorial 3: Train a Simple Regression Model
   :width: 160px
   :height: 100px

.. image:: ../_static/img/tutorial_4_video.png
   :target: https://www.youtube.com/watch?v=El5ZcyDRMhQ
   :alt: Tutorial 4: Application of a Trained Model
   :width: 160px
   :height: 100px

|

.. image:: ../_static/img/tutorial_5_video.png
   :target: https://www.youtube.com/watch?v=lK-pP3mZAng
   :alt: Tutorial 5: Hyperparameter Optimization
   :width: 160px
   :height: 100px

.. image:: ../_static/img/tutorial_6_video.png
   :target: https://www.youtube.com/watch?v=fNdSZGtZjWk
   :alt: Tutorial 6: Comparing models to select the best hyperparameters
   :width: 160px
   :height: 100px

.. image:: ../_static/img/tutorial_7_video.png
   :target: https://www.youtube.com/watch?v=uC7aNILqnCc
   :alt: Tutorial 7: Train a Production Model
   :width: 160px
   :height: 100px

.. image:: ../_static/img/tutorial_8_video.png
   :target: https://www.youtube.com/watch?v=D29yObV8AYI
   :alt: Tutorial 8: Visualizations of Model Performance
   :width: 160px
   :height: 100px

How to Use These Tutorials
**************************

We have provided the AMPL tutorials in the readthedocs and as Jupyter notebooks available on our GitHub. 
Depending on your knowledge level and preferred learning style, you can use these tutorials in any of several 
ways:

*	Run the Jupyter notebooks after installing AMPL on your local system. You will find the notebooks under "atomsci/ddm/examples/tutorials". If you use the Docker installation option described in our Docker installation instructions, you will find tutorials under "/AMPL/atomsci/ddm/examples/tutorials".
*	Create a new notebook and type in (or copy and paste) the Python code from the tutorials yourself. Some people prefer this method because it helps them learn better. If you take this route,  create your notebook in the `tutorials` directory, or start it with a pair of commands like the following , so that the tutorial code can find the data files we’ve provided under the "tutorials" directory:

  .. code::

      import os
      os.chdir(“my_ampl_root/atomsci/ddm/examples/tutorials”)

*	Read the tutorial pages here, then go straight to writing your own `AMPL <https://github.com/ATOMScience-org/AMPL>`_ application  to create models from your own data. We won’t hold you back!
*	You are also free to modify the tutorials as you wish to try out different model parameters or apply the techniques to your own data. Just be aware that, if you are running the tutorials in a Docker container, any changes you make in the `tutorials` directory will be lost when you shut down the container. The installation instructions offer some suggestions for saving your work in this scenario.
 
Although the tutorials are designed to be run in sequence, using an example dataset (SLC6A3: small molecule inhibition of the dopamine transporter) 
provided within `AMPL <https://github.com/ATOMScience-org/AMPL>`_, we have also provided copies of the intermediate files generated by each tutorial that are 
required by subsequent tutorials, so that you can run them in any order.
 
Also, if you have issues or questions about the tutorials, please create an issue `here <https://github.com/ATOMScience-org/AMPL/issues>`_.
