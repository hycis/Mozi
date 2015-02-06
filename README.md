Pynet
=====

<dl class="method">
<dt id="pylearn2.models.rbm.GaussianBinaryRBM.free_energy">
<tt class="descname">free_energy</tt><big>(</big><em>V</em><big>)</big><a class="headerlink" href="#pylearn2.models.rbm.GaussianBinaryRBM.free_energy" title="Permalink to this definition">¶</a></dt>
<dd><div class="admonition-todo admonition" id="index-17">
<p class="first admonition-title">Todo</p>
<p class="last">WRITEME</p>
</div>
</dd></dl>

<dl class="method">
<dt id="pylearn2.models.model.Model.get_default_cost">
<tt class="descname">get_default_cost</tt><big>(</big><big>)</big><a class="headerlink" href="#pylearn2.models.model.Model.get_default_cost" title="Permalink to this definition">¶</a></dt>
<dd><p>Returns the default cost to use with this model.</p>
<table class="docutils field-list" frame="void" rules="none">
<col class="field-name" />
<col class="field-body" />
<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Returns :</th><td class="field-body"><p class="first"><strong>default_cost</strong> : Cost</p>
<blockquote class="last">
<div><p>The default cost to use with this model.</p>
</div></blockquote>
</td>
</tr>
</tbody>
</table>
</dd></dl>
Pynet is meant to be a simple and straight forward framework base on Theano, it aims to be a much cleaner and focused framework than pylearn2, and only aims to do the best for neural network computation. It is for anyone who wants to run large dataset on GPU with 10X speedup, and who has a tough time learning the much more bulky pylearn2.

Pynet has been used to reproduce many state-of-the-art results, such as dropout and maxout on mnist And it's a stable and fast.

Pynet consists of the following modules

1. [TrainObject](doc/train_object.md)
2. [MLP](doc/model.md)
3. [Dataset](doc/dataset.md)
4. [Learning Method](doc/learning_method.md)
5. [Layer](doc/layer.md)
6. [Learning Rule](doc/learning_rule.md)
7. [Log](doc/log.md)

__Get Started with Simple Example__

You can start with a simple [MLP example](doc/mlp_example.md)
Or start with a simple [AutoEncoder example](doc/ae_example.md)
