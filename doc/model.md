
<div class="section" id="model-object">

<p>A model object is used to put together the layers, a model provides an abstract class for building an autoencoder or an mlp.</p>
<dl class="class">
<dt id="model.Model">
<em class="property">class </em><tt class="descclassname">model.Model</tt><big>(</big><em>input_dim</em>, <em>rand_seed=None</em><big>)</big></dt>
<dd><p>An interface for the MLP and the autoencoder class</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>input_dim</strong> : int</p>
<blockquote>
<div><p>Input dimension to the model</p>
</div></blockquote>
<p><strong>rand_seed</strong> : int</p>
<blockquote>
<div><p>This is used to set the initialization of all the weights in the network</p>
</div></blockquote>
</td>
</tr>

</tbody>



</dd></dl>



<div class="section" id="mlp-object">
<dl class="class">
<dt id="model.MLP">
<em class="property">class </em><tt class="descclassname">model.MLP</tt><big>(</big><em>input_dim</em>, <em>rand_seed=None</em><big>)</big></dt>
<dd><p>This is used to build a feedforward neural network</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>input_dim</strong> : int</p>
<blockquote>
<div><p>Input dimension to the model</p>
</div></blockquote>
<p><strong>rand_seed</strong> : int</p>
<blockquote>
<div><p>This is used to set the initialization of all the weights in the network</p>
</div></blockquote>
</td>
</tr>
</tbody>


<dl class="instance method">
<dt id="model.MLP.add_layer">
<tt class="descclassname">add_layer</tt><big>(</big><em>layer</em><big>)</big></dt>
<dd><p>This is used to add layer to the model</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>layer</strong> : layer instance</p>
</td>
</tr>
</tbody>
</dd></dl>


<dl class="instance method">
<dt id="model.MLP.pop_layer">
<tt class="descclassname">pop_layer</tt><big>(</big><em>index</em><big>)</big></dt>
<dd><p>This is used to pop a layer from the model</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>layer</strong> : index</p>
  <blockquote>
  <div><p>The index of the layer to be popped</p>
  </div></blockquote>
</td>
</tr>
</tbody>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Return:</th><td class="field-body">
  <blockquote>
  <div><p>Return layer</p>
  </div></blockquote>
</td>
</tr>
</tbody>

</dd></dl>

<dl class="instance method">
<dt id="model.MLP.fprop">
<tt class="descclassname"> fprop</tt><big>(</big><em>input_values</em><big>)</big></dt>
<dd><p>This is used to forward propagate the input values through the mlp</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>input_values</strong> : 2d numpy array</p>
  <blockquote>
  <div><p>the input value is X which is a two dimensional numpy of dimension (num_examples, num_features)</p>
  </div></blockquote>
</td>
</tr>
</tbody>
</dd></dl>


<dl class="instance method">
<dt id=" get_layers">
<tt class="descclassname"> get_layers</tt><big>(</big><big>)</big></dt>
<dd><p>Return the layers in the mlp</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Return:</th><td class="field-body">
  <blockquote>
  <div><p>Return a list of all the layers</p>
  </div></blockquote>
</td>
</tr>
</tbody>
</dd></dl>

</dd></dl>




<div class="section" id="ae-object">
<!-- <span id="datetime-date"></span><h2>8.1.3. <a class="reference internal" href="#datetime.date" title="datetime.date"><tt class="xref py py-class docutils literal"><span class="pre">date</span></tt></a> Objects<a class="headerlink" href="#date-objects" title="Permalink to this headline">¶</a></h2>
<p>A model object is used to put together the layers, a model provides an abstract class for building an autoencoder or an mlp.</p> -->
<dl class="class">
<dt id="model.AutoEncoder">
<em class="property">class </em><tt class="descclassname">model.AutoEncoder</tt><big>(</big><em>input_dim</em>, <em>rand_seed=None</em><big>)</big></dt>
<dd><p>This is used to build an autoencoder</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>input_dim</strong> : int</p>
<blockquote>
<div><p>Input dimension to the model</p>
</div></blockquote>
<p><strong>rand_seed</strong> : int</p>
<blockquote>
<div><p>This is used to set the initialization of all the weights in the network</p>
</div></blockquote>
</td>
</tr>
</tbody>

<!-- <p>Example of creating a model</p>
<div class="highlight-python" style="position: relative;"><div class="highlight"><span class="copybutton" title="Hide the prompts and output" style="cursor: pointer; position: absolute; top: 0px; right: 0px; border: 1px solid rgb(170, 204, 153); color: rgb(170, 204, 153); font-family: monospace; padding-left: 0.2em; padding-right: 0.2em;"></span><pre>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from pynet.model import MLP</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn"></span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
</pre></div>
</div> -->

<dl class="instance method">
<dt id="model.AutoEncoder.add_encode_layer">
<tt class="descclassname">add_encode_layer</tt><big>(</big><em>layer</em><big>)</big></dt>
<dd><p>Add to the encode layers</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>layer</strong> : layer instance</p>
</td>
</tr>
</tbody>
</dd></dl>


<dl class="instance method">
<dt id="model.AutoEncoder.add_decode_layer">
<tt class="descclassname">add_decode_layer</tt><big>(</big><em>layer</em><big>)</big></dt>
<dd><p>Add to the decode layers</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>layer</strong> : layer instance</p>
</td>
</tr>
</tbody>
</dd></dl>


<dl class="instance method">
<dt id="model.AutoEncoder.pop_encode_layer">
<tt class="descclassname">pop_encode_layer</tt><big>(</big><em>index=0</em><big>)</big></dt>
<dd><p>Pop the layer from the encode layers</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>layer</strong> : index</p>
  <blockquote>
  <div><p>The index of the encode layer to be popped</p>
  </div></blockquote>
</td>
</tr>
</tbody>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Return:</th><td class="field-body">
  <blockquote>
  <div><p>Return the popped layer</p>
  </div></blockquote>
</td>
</tr>
</tbody>

</dd></dl>


<dl class="instance method">
<dt id="model.AutoEncoder.pop_decode_layer">
<tt class="descclassname">pop_decode_layer</tt><big>(</big><em>index=0</em><big>)</big></dt>
<dd><p>Pop the layer from the decode layers</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>layer</strong> : index</p>
  <blockquote>
  <div><p>The index of the decode layer to be popped</p>
  </div></blockquote>
</td>
</tr>
</tbody>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Return:</th><td class="field-body">
  <blockquote>
  <div><p>Return the popped layer</p>
  </div></blockquote>
</td>
</tr>
</tbody>

</dd></dl>


<dl class="instance method">
<dt id="model.AutoEncoder.encode">
<tt class="descclassname">encode</tt><big>(</big><em>input_values</em><big>)</big></dt>
<dd><p>This is used to forward propagate the input values through the encode layers</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>input_values</strong> : 2d numpy array</p>
  <blockquote>
  <div><p>the input value is a two dimensional numpy of dimension (num_examples, input_dim)</p>
  </div></blockquote>
</td>
</tr>
</tbody>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Return:</th><td class="field-body">
  <blockquote>
  <div><p>Return 2d numpy array of dimension (num_examples, bottleneck_dim) where bottleneck_dim is the dimension of the bottleneck in an autoencoder</p>
  </div></blockquote>
</td>
</tr>
</tbody>
</dd></dl>


<dl class="instance method">
<dt id="model.AutoEncoder.decode">
<tt class="descclassname">decode</tt><big>(</big><em>input_values</em><big>)</big></dt>
<dd><p>This is used to forward propagate the input values through the decode layers</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>input_values</strong> : 2d numpy array</p>
  <blockquote>
  <div><p>the input value is a two dimensional numpy of dimension (num_examples, bottleneck_dim) where bottleneck_dim is the dimension of the bottleneck in an autoencoder</p>
  </div></blockquote>
</td>
</tr>
</tbody>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Return:</th><td class="field-body">
  <blockquote>
  <div><p>Return 2d numpy array of dimension (num_examples, output_dim)</p>
  </div></blockquote>
</td>
</tr>
</tbody>
</dd></dl>




</dd></dl>



</div>
