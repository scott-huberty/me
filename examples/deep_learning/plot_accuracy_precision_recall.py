"""
Accuracy, precision, and recall
===============================

.. figure:: /_static/imgs/accuracy.png

Accuracy
--------

Formula
^^^^^^^

.. math::
    \\dfrac{\\text{True Positives} + \\text{True Negatives}}{\\text{Total number of samples}}


Interpretation
^^^^^^^^^^^^^^

The proportion of correctly classified samples. It is an overall indicator
of the model's performance (irrespective of sample sizes and class/category (im)balance).

.. caution::

    Accuracy ignores possible biases introduced by unbalanced sample sizes. I.e. it
    ignores the "two ways of being wrong" problem (see below).

Precision
---------

Formula
^^^^^^^

.. math::

    \\dfrac{\\text{True Positives}}{\\text{Total number of "yes" predictions}}

Interpretation
^^^^^^^^^^^^^^

Reveals when the model has a bias towards saying "yes". Precision includes a penalty
for misclassifying negative samples as positive. This is useful when the cost of
misclassifying a negative sample as positive is high. For example, in fraud detection.

Recall (aka Sensitivity)
------------------------

Formula
^^^^^^^

.. math::

    \\dfrac{\\text{True Positives}}{\\text{Total number of actual positive samples}}

interpretation
^^^^^^^^^^^^^^

Reveals when the model has a bias for saying "no". It includes a penalty for false
negatives. This is useful when the cost of classifying a positive sample as
negative is high, for example, in cancer detection.


F1
--

The F1 score basically incorporates the 3 metrics above into a single metric.

Formula
^^^^^^^

.. math::

    \\dfrac{ \\text{True Positives} }{ \\text{True Positives} + \\dfrac{1}{2} (\\text{False Positives} + \\text{False Negatives}) }

The denominator is True Positives plus the average of the "two ways of being wrong".

Interpretation
^^^^^^^^^^^^^^

The F1 provides a balance between precision and recall. It gives a general idea of
whether the model is biased (but doesn't tell you which way it is biased). An F1
score is high when the model makes few mistakes.
"""