/******************************************************************************

Copyright (c) 2015 by SAS Institute Inc., Cary, NC 27513 USA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

******************************************************************************/

******************************************************************************;
* VARIOUS SAS ROUTINES FOR APPROXIMATING A STACKED AUTOENCODER:              *;
* CREATE LIST OF INPUTS AS A MACRO                                           *;
* STANDARDIZE NUMERIC INPUTS - REQUIRED BEFORE CLUSTERING                    *;
* CREATE 16 K-MEANS CLUSTERS                                                 *;
* CREATE DATA MINING DATABASE WITH PROC DMDB -                               *;
*   REQUIRED CATALOG FOR PROC NEURAL                                         *;
* TRAIN SIMPLE STACKED AUTOENCODER WITH 5 HIDDEN LAYERS                      *;
* SCORE TRAINING DATA WITH TRAINED NEURAL NETWORK                            *;
* KEEP OUTPUT OF 2-DIMENSIONAL MIDDLE LAYER (H31,H32) AS NEW FEATURE SPACE   *;
* SET THE MEDIAN POINT AS THE ORIGIN OF THE NEW 2-D FEATURE SPACE            *;
* CALCULATE THE DISTANCE OF EACH POINT IN NEW FEATURE SPACE FROM THIS ORIGIN *;
* DETECT OUTLIERS AS FIRST 5 POINTS FARTHEST FROM THE ORIGIN                 *;
* DISPLAY THE DATA IN 2-D INCLUDING CLUSTER LABELS AND OUTLIERS              *;
*                                                                            *;
* RELATED TO:                                                                *;
* HINTON, G. E. AND SALAKHUTDINOV, R. R. 2006. “REDUCING THE DIMENSIONALITY  *;
* OF DATA WITH NEURAL NETWORKS.” SCIENCE 313:504–507.                        *;
*                                                                            *;
* VINCENT, PASCAL, ET AL. "EXTRACTING AND COMPOSING ROBUST FEATURES WITH     *;
* DENOISING AUTOENCODERS." PROCEEDINGS OF THE 25TH INTERNATIONAL CONFERENCE  *;
* ON MACHINE LEARNING. ACM, 2008.                                            *;
******************************************************************************;

******************************************************************************;
* BASIC SYSTEM AND METADATA SETUP                                            *;
******************************************************************************;

*** SET GITHUB DATA DIRECTORY (MUST CONTAIN PROVIDER_SUMMARY.SAS7BDAT);
%let git_repo_dir = ;

*** SET NUMBER OF AVAILABLE CORES;
%let num_cores = ;

*** SET THE NUMBER OF OUTLIERS TO FIND (DEFAULT=5);
%let num_outliers=5;

x cd "&git_repo_dir.";
libname l "&git_repo_dir.";
data provider_summary;
   set l.provider_summary;
run;

ods listing;
ods html close;

*** PLACE NUMERIC INPUTS INTO MACRO LIST FOR CONVENIENCE;
proc contents
   data=provider_summary
   out=names (keep=name
      where=(strip(name)^="provider_id"
      and strip(name)^="name"));
run;
filename emutil catalog 'sashelp.emutil.em_varmacro.source';
%include emutil;
filename emutil;
%EM_VARMACRO(
   name=INPUTS,
   metadata=names,
   nummacro=NUM_INPUTS
);

******************************************************************************;
* K-MEANS CLUSTERING                                                         *;
******************************************************************************;

*** STANDARDIZE NUMERIC INPUTS FOR K-MEANS CLUSTERING;
proc stdize
   data=provider_summary
   out=std_provider_summary(drop=name)
   method=std;
   var %INPUTS;
run;

*** CLUSTERING;
proc fastclus
   data=std_provider_summary
   maxclusters=16
   maxiter=100
   out=outc (drop=distance);
   var %INPUTS;
run;

******************************************************************************;
* TRAIN SIMPLE STACKED AUTOENCODER AND SCORE TRAINING DATA                   *;
******************************************************************************;

*** CREATE DATA MINING DATABASE WITH PROC DMDB - REQUIRED FOR PROC NEURAL;
proc dmdb
   data=outc
   out=outc_dmdb
   dmdbcat=work.cat_outc_dmdb;
   var %INPUTS;
   id cluster;
   target %INPUTS;
run;

*** TRAIN SIMPLE STACKED AUTOENCODER WITH 5 HIDDEN LAYERS;
proc neural
   data=outc
   dmdbcat=work.cat_outc_dmdb
   random=44444;
   performance compile details cpucount=&num_cores. threads=yes;

   nloptions fconv=0.00001 noprint; /* noprint=DO NOT SHOW WEIGHT VALUES */
   netoptions decay=1.0;
   /* DENOISING, JITTER */
   /* L2 PENALTY ROUGHLY EQUIVALENT TO GAUSSIAN NOISE INJECTION */

   /* 5-LAYER NETWORK ARCHITECTURE */
   archi MLP hidden=5;
   hidden &NUM_INPUTS / id=h1;
   hidden %eval(&NUM_INPUTS/2) / id=h2;
   hidden 2 / id=h3 act=linear;
   hidden %eval(&NUM_INPUTS/2) / id=h4;
   hidden &NUM_INPUTS / id=h5;
   input %INPUTS / std=no id=i level=int;
   target %INPUTS / std=no id=t level=int;

   /* INTIALIZE */
   initial infan=1; /* INFAN REDUCES SATURATION OF NEURONS BY RANDOM INIT */
   prelim 20 preiter=20;

   /* TRAIN LAYERS SEPARATELY */
   freeze h1->h2;
   freeze h2->h3;
   freeze h3->h4;
   freeze h4->h5;
   train
   	maxtime=10000
   	maxiter=5000
   	outest=weights_layer1
   	estiter=1;

   freeze i->h1;
   thaw h1->h2;
   train
   	maxtime=10000
   	maxiter=5000
   	outest=weights_layer2
   	estiter=1;

   freeze h1->h2;
   thaw h2->h3;
   train
   	maxtime=10000
   	maxiter=5000
   	outest=weights_layer3
   	estiter=1;

   freeze h2->h3;
   thaw h3->h4;
   train
   	maxtime=10000
   	maxiter=5000
   	outest=weights_layer4
   	estiter=1;

   freeze h3->h4;
   thaw h4->h5;
   train
   	maxtime=10000
   	maxiter=5000
   	outest=weights_layer5
   	estiter=1;

   /* RETRAIN ALL LAYERS SIMULTANEOUSLY */
   thaw i->h1;
   thaw h1->h2;
   thaw h2->h3;
   thaw h3->h4;
   train
   	maxtime=10000
   	maxiter=5000
   	outest=weights_all
   	estiter=1;

   /* CREATE PORTABLE SCORE CODE */
   code file="autoencoder.sas";

run;

*** SCORE TRAINING DATA, RETAINING OUTPUT FROM THE MIDDLE HIDDEN WEIGHTS;
data score2D(keep=h31 h32 provider_id);
   set outc;
   %include "autoencoder.sas";
run;

******************************************************************************;
* CREATE ORIGIN POINT AND FIND ALL EUCLIDEAN DISTANCES FROM IT               *;
******************************************************************************;

*** SET THE MEDIAN POINT AS ORIGIN OF NEW FEATURE SPACE;
proc means data=score2D;
   output out=origin median(h31 h32)=h31 h32;
run;
data origin;
   set origin(keep=h31 h32);
   provider_id=0;
run;

*** STACK THE ORIGIN AS LAST ROW;
data score2D_origin;
   set score2D origin;
run;

*** FIND PAIRWISE DISTANCES FOR EVERY POINT INCLUDING ORIGIN;
proc distance
   data=score2D_origin
   out=distance
   method=euclid
   shape=square
   nostd;
   var interval (h31 h32);
   copy provider_id;
run;

*** KEEP ONLY THE LAST COLUMN (Dist3338: DISTANCE TO ORIGIN);
data distance_to_origin (keep=Dist3338 provider_id);
   set distance;
   if Dist3338=0 then delete; /* DISTANCE FROM SELF IS IRRELEVANT */
run;

******************************************************************************;
* MAKE A PRETTY PICTURE                                                      *;
******************************************************************************;

*** GET PROVIDER NAMES;
data provider_names (keep=provider_id name);
   set provider_summary;
run;

*** SORT DATA SETS FOR MERGING;
proc sort
   data=provider_names;
   by provider_id;
run;
proc sort
   data=score2d (keep=h31 h32 provider_id);
   by provider_id;
run;
proc sort data=distance_to_origin;
   by provider_id;
run;

*** MERGE DATA FOR PLOTTING;
data plot;
   merge provider_names outc(keep=provider_id cluster) score2d
      distance_to_origin;
   by provider_id;
run;

*** DETECT THE OUTLIERS AS FIRST NUM_OUTLIERS POINTS;
*** FARTHEST FROM THE NEW ORIGIN;
proc sort data=plot; by descending Dist3338; run;
data plot;
   set plot;
   if _n_> &num_outliers. then name='';
run;

*** PLOT;
ods graphics / labelmax=3400; /* LABEL COLLISION AVOIDANCE */
proc sgplot data=plot;
   scatter x=h31 y=h32 /
   transparency=0.35
   group=cluster
   markerattrs=(size=9 symbol=circleFilled)
   datalabel=name
   nomissinggroup;
run;
