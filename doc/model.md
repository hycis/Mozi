
<div class="section" id="date-objects">
<span id="datetime-date"></span><h2>8.1.3. <a class="reference internal" href="#datetime.date" title="datetime.date"><tt class="xref py py-class docutils literal"><span class="pre">date</span></tt></a> Objects<a class="headerlink" href="#date-objects" title="Permalink to this headline">¶</a></h2>
<p>A model object is used to put together the layers, a model provides an abstract class for building an autoencoder or an mlp.</p>
<dl class="class">
<dt id="model.Model">
<em class="property">class </em><tt class="descclassname">model.</tt><tt class="descname">Model</tt><big>(</big><em>input_dim</em>, <em>rand_seed=None</em><big>)</big></dt>
<dd><p>An interface for the MLP and the autoencoder class</p>
<p>If an argument outside those ranges is given, <a class="reference internal" href="exceptions.html#exceptions.ValueError" title="exceptions.ValueError"><tt class="xref py py-exc docutils literal"><span class="pre">ValueError</span></tt></a> is raised.</p>

<tbody valign="top">
<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><p class="first"><strong>a</strong> : array_like</p>
<blockquote>
<div><p>Input array.</p>
</div></blockquote>
<p><strong>axes</strong> : list of ints, optional</p>
<blockquote>
<div><p>By default, reverse the dimensions, otherwise permute the axes
according to the values given.</p>
</div></blockquote>
</td>
</tr>
<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>p</strong> : ndarray</p>
<blockquote class="last">
<div><p><em class="xref py py-obj">a</em> with its axes permuted.  A view is returned whenever
possible.</p>
</div></blockquote>
</td>
</tr>
</tbody>


<p>Example of counting days to an event:</p>
<div class="highlight-python" style="position: relative;"><div class="highlight"><span class="copybutton" title="Hide the prompts and output" style="cursor: pointer; position: absolute; top: 0px; right: 0px; border: 1px solid rgb(170, 204, 153); color: rgb(170, 204, 153); font-family: monospace; padding-left: 0.2em; padding-right: 0.2em;">&gt;&gt;&gt;</span><pre>

<span class="gp">&gt;&gt;&gt; </span><span class="kn">import time</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
</pre></div>
</div>


</dd></dl>


<p>Other constructors, all class methods:</p>
<dl class="classmethod">
<dt id="datetime.date.today">
<em class="property">classmethod </em><tt class="descclassname">date.</tt><tt class="descname">today</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.today" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the current local date.  This is equivalent to
<tt class="docutils literal"><span class="pre">date.fromtimestamp(time.time())</span></tt>.</p>
</dd></dl>

<dl class="classmethod">
<dt id="datetime.date.fromtimestamp">
<em class="property">classmethod </em><tt class="descclassname">date.</tt><tt class="descname">fromtimestamp</tt><big>(</big><em>timestamp</em><big>)</big><a class="headerlink" href="#datetime.date.fromtimestamp" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the local date corresponding to the POSIX timestamp, such as is returned
by <a class="reference internal" href="time.html#time.time" title="time.time"><tt class="xref py py-func docutils literal"><span class="pre">time.time()</span></tt></a>.  This may raise <a class="reference internal" href="exceptions.html#exceptions.ValueError" title="exceptions.ValueError"><tt class="xref py py-exc docutils literal"><span class="pre">ValueError</span></tt></a>, if the timestamp is out
of the range of values supported by the platform C <tt class="xref c c-func docutils literal"><span class="pre">localtime()</span></tt> function.
It’s common for this to be restricted to years from 1970 through 2038.  Note
that on non-POSIX systems that include leap seconds in their notion of a
timestamp, leap seconds are ignored by <a class="reference internal" href="#datetime.date.fromtimestamp" title="datetime.date.fromtimestamp"><tt class="xref py py-meth docutils literal"><span class="pre">fromtimestamp()</span></tt></a>.</p>
</dd></dl>

<dl class="classmethod">
<dt id="datetime.date.fromordinal">
<em class="property">classmethod </em><tt class="descclassname">date.</tt><tt class="descname">fromordinal</tt><big>(</big><em>ordinal</em><big>)</big><a class="headerlink" href="#datetime.date.fromordinal" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the date corresponding to the proleptic Gregorian ordinal, where January
1 of year 1 has ordinal 1.  <a class="reference internal" href="exceptions.html#exceptions.ValueError" title="exceptions.ValueError"><tt class="xref py py-exc docutils literal"><span class="pre">ValueError</span></tt></a> is raised unless <tt class="docutils literal"><span class="pre">1</span> <span class="pre">&lt;=</span> <span class="pre">ordinal</span> <span class="pre">&lt;=</span>
<span class="pre">date.max.toordinal()</span></tt>. For any date <em>d</em>, <tt class="docutils literal"><span class="pre">date.fromordinal(d.toordinal())</span> <span class="pre">==</span>
<span class="pre">d</span></tt>.</p>
</dd></dl>

<p>Class attributes:</p>
<dl class="attribute">
<dt id="datetime.date.min">
<tt class="descclassname">date.</tt><tt class="descname">min</tt><a class="headerlink" href="#datetime.date.min" title="Permalink to this definition">¶</a></dt>
<dd><p>The earliest representable date, <tt class="docutils literal"><span class="pre">date(MINYEAR,</span> <span class="pre">1,</span> <span class="pre">1)</span></tt>.</p>
</dd></dl>

<dl class="attribute">
<dt id="datetime.date.max">
<tt class="descclassname">date.</tt><tt class="descname">max</tt><a class="headerlink" href="#datetime.date.max" title="Permalink to this definition">¶</a></dt>
<dd><p>The latest representable date, <tt class="docutils literal"><span class="pre">date(MAXYEAR,</span> <span class="pre">12,</span> <span class="pre">31)</span></tt>.</p>
</dd></dl>

<dl class="attribute">
<dt id="datetime.date.resolution">
<tt class="descclassname">date.</tt><tt class="descname">resolution</tt><a class="headerlink" href="#datetime.date.resolution" title="Permalink to this definition">¶</a></dt>
<dd><p>The smallest possible difference between non-equal date objects,
<tt class="docutils literal"><span class="pre">timedelta(days=1)</span></tt>.</p>
</dd></dl>

<p>Instance attributes (read-only):</p>
<dl class="attribute">
<dt id="datetime.date.year">
<tt class="descclassname">date.</tt><tt class="descname">year</tt><a class="headerlink" href="#datetime.date.year" title="Permalink to this definition">¶</a></dt>
<dd><p>Between <a class="reference internal" href="#datetime.MINYEAR" title="datetime.MINYEAR"><tt class="xref py py-const docutils literal"><span class="pre">MINYEAR</span></tt></a> and <a class="reference internal" href="#datetime.MAXYEAR" title="datetime.MAXYEAR"><tt class="xref py py-const docutils literal"><span class="pre">MAXYEAR</span></tt></a> inclusive.</p>
</dd></dl>

<dl class="attribute">
<dt id="datetime.date.month">
<tt class="descclassname">date.</tt><tt class="descname">month</tt><a class="headerlink" href="#datetime.date.month" title="Permalink to this definition">¶</a></dt>
<dd><p>Between 1 and 12 inclusive.</p>
</dd></dl>

<dl class="attribute">
<dt id="datetime.date.day">
<tt class="descclassname">date.</tt><tt class="descname">day</tt><a class="headerlink" href="#datetime.date.day" title="Permalink to this definition">¶</a></dt>
<dd><p>Between 1 and the number of days in the given month of the given year.</p>
</dd></dl>

<p>Supported operations:</p>
<table border="1" class="docutils">
<colgroup>
<col width="40%">
<col width="60%">
</colgroup>
<thead valign="bottom">
<tr class="row-odd"><th class="head">Operation</th>
<th class="head">Result</th>
</tr>
</thead>
<tbody valign="top">
<tr class="row-even"><td><tt class="docutils literal"><span class="pre">date2</span> <span class="pre">=</span> <span class="pre">date1</span> <span class="pre">+</span> <span class="pre">timedelta</span></tt></td>
<td><em>date2</em> is <tt class="docutils literal"><span class="pre">timedelta.days</span></tt> days removed
from <em>date1</em>.  (1)</td>
</tr>
<tr class="row-odd"><td><tt class="docutils literal"><span class="pre">date2</span> <span class="pre">=</span> <span class="pre">date1</span> <span class="pre">-</span> <span class="pre">timedelta</span></tt></td>
<td>Computes <em>date2</em> such that <tt class="docutils literal"><span class="pre">date2</span> <span class="pre">+</span>
<span class="pre">timedelta</span> <span class="pre">==</span> <span class="pre">date1</span></tt>. (2)</td>
</tr>
<tr class="row-even"><td><tt class="docutils literal"><span class="pre">timedelta</span> <span class="pre">=</span> <span class="pre">date1</span> <span class="pre">-</span> <span class="pre">date2</span></tt></td>
<td>(3)</td>
</tr>
<tr class="row-odd"><td><tt class="docutils literal"><span class="pre">date1</span> <span class="pre">&lt;</span> <span class="pre">date2</span></tt></td>
<td><em>date1</em> is considered less than <em>date2</em> when
<em>date1</em> precedes <em>date2</em> in time. (4)</td>
</tr>
</tbody>
</table>
<p>Notes:</p>
<ol class="arabic simple">
<li><em>date2</em> is moved forward in time if <tt class="docutils literal"><span class="pre">timedelta.days</span> <span class="pre">&gt;</span> <span class="pre">0</span></tt>, or backward if
<tt class="docutils literal"><span class="pre">timedelta.days</span> <span class="pre">&lt;</span> <span class="pre">0</span></tt>.  Afterward <tt class="docutils literal"><span class="pre">date2</span> <span class="pre">-</span> <span class="pre">date1</span> <span class="pre">==</span> <span class="pre">timedelta.days</span></tt>.
<tt class="docutils literal"><span class="pre">timedelta.seconds</span></tt> and <tt class="docutils literal"><span class="pre">timedelta.microseconds</span></tt> are ignored.
<a class="reference internal" href="exceptions.html#exceptions.OverflowError" title="exceptions.OverflowError"><tt class="xref py py-exc docutils literal"><span class="pre">OverflowError</span></tt></a> is raised if <tt class="docutils literal"><span class="pre">date2.year</span></tt> would be smaller than
<a class="reference internal" href="#datetime.MINYEAR" title="datetime.MINYEAR"><tt class="xref py py-const docutils literal"><span class="pre">MINYEAR</span></tt></a> or larger than <a class="reference internal" href="#datetime.MAXYEAR" title="datetime.MAXYEAR"><tt class="xref py py-const docutils literal"><span class="pre">MAXYEAR</span></tt></a>.</li>
<li>This isn’t quite equivalent to date1 + (-timedelta), because -timedelta in
isolation can overflow in cases where date1 - timedelta does not.
<tt class="docutils literal"><span class="pre">timedelta.seconds</span></tt> and <tt class="docutils literal"><span class="pre">timedelta.microseconds</span></tt> are ignored.</li>
<li>This is exact, and cannot overflow.  timedelta.seconds and
timedelta.microseconds are 0, and date2 + timedelta == date1 after.</li>
<li>In other words, <tt class="docutils literal"><span class="pre">date1</span> <span class="pre">&lt;</span> <span class="pre">date2</span></tt> if and only if <tt class="docutils literal"><span class="pre">date1.toordinal()</span> <span class="pre">&lt;</span>
<span class="pre">date2.toordinal()</span></tt>. In order to stop comparison from falling back to the
default scheme of comparing object addresses, date comparison normally raises
<a class="reference internal" href="exceptions.html#exceptions.TypeError" title="exceptions.TypeError"><tt class="xref py py-exc docutils literal"><span class="pre">TypeError</span></tt></a> if the other comparand isn’t also a <a class="reference internal" href="#datetime.date" title="datetime.date"><tt class="xref py py-class docutils literal"><span class="pre">date</span></tt></a> object.
However, <tt class="docutils literal"><span class="pre">NotImplemented</span></tt> is returned instead if the other comparand has a
<tt class="xref py py-meth docutils literal"><span class="pre">timetuple()</span></tt> attribute.  This hook gives other kinds of date objects a
chance at implementing mixed-type comparison. If not, when a <a class="reference internal" href="#datetime.date" title="datetime.date"><tt class="xref py py-class docutils literal"><span class="pre">date</span></tt></a>
object is compared to an object of a different type, <a class="reference internal" href="exceptions.html#exceptions.TypeError" title="exceptions.TypeError"><tt class="xref py py-exc docutils literal"><span class="pre">TypeError</span></tt></a> is raised
unless the comparison is <tt class="docutils literal"><span class="pre">==</span></tt> or <tt class="docutils literal"><span class="pre">!=</span></tt>.  The latter cases return
<a class="reference internal" href="constants.html#False" title="False"><tt class="xref py py-const docutils literal"><span class="pre">False</span></tt></a> or <a class="reference internal" href="constants.html#True" title="True"><tt class="xref py py-const docutils literal"><span class="pre">True</span></tt></a>, respectively.</li>
</ol>
<p>Dates can be used as dictionary keys. In Boolean contexts, all <a class="reference internal" href="#datetime.date" title="datetime.date"><tt class="xref py py-class docutils literal"><span class="pre">date</span></tt></a>
objects are considered to be true.</p>
<p>Instance methods:</p>
<dl class="method">
<dt id="datetime.date.replace">
<tt class="descclassname">date.</tt><tt class="descname">replace</tt><big>(</big><em>year</em>, <em>month</em>, <em>day</em><big>)</big><a class="headerlink" href="#datetime.date.replace" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a date with the same value, except for those parameters given new
values by whichever keyword arguments are specified.  For example, if <tt class="docutils literal"><span class="pre">d</span> <span class="pre">==</span>
<span class="pre">date(2002,</span> <span class="pre">12,</span> <span class="pre">31)</span></tt>, then <tt class="docutils literal"><span class="pre">d.replace(day=26)</span> <span class="pre">==</span> <span class="pre">date(2002,</span> <span class="pre">12,</span> <span class="pre">26)</span></tt>.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.timetuple">
<tt class="descclassname">date.</tt><tt class="descname">timetuple</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.timetuple" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a <a class="reference internal" href="time.html#time.struct_time" title="time.struct_time"><tt class="xref py py-class docutils literal"><span class="pre">time.struct_time</span></tt></a> such as returned by <a class="reference internal" href="time.html#time.localtime" title="time.localtime"><tt class="xref py py-func docutils literal"><span class="pre">time.localtime()</span></tt></a>.
The hours, minutes and seconds are 0, and the DST flag is -1. <tt class="docutils literal"><span class="pre">d.timetuple()</span></tt>
is equivalent to <tt class="docutils literal"><span class="pre">time.struct_time((d.year,</span> <span class="pre">d.month,</span> <span class="pre">d.day,</span> <span class="pre">0,</span> <span class="pre">0,</span> <span class="pre">0,</span>
<span class="pre">d.weekday(),</span> <span class="pre">yday,</span> <span class="pre">-1))</span></tt>, where <tt class="docutils literal"><span class="pre">yday</span> <span class="pre">=</span> <span class="pre">d.toordinal()</span> <span class="pre">-</span> <span class="pre">date(d.year,</span> <span class="pre">1,</span>
<span class="pre">1).toordinal()</span> <span class="pre">+</span> <span class="pre">1</span></tt> is the day number within the current year starting with
<tt class="docutils literal"><span class="pre">1</span></tt> for January 1st.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.toordinal">
<tt class="descclassname">date.</tt><tt class="descname">toordinal</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.toordinal" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the proleptic Gregorian ordinal of the date, where January 1 of year 1
has ordinal 1.  For any <a class="reference internal" href="#datetime.date" title="datetime.date"><tt class="xref py py-class docutils literal"><span class="pre">date</span></tt></a> object <em>d</em>,
<tt class="docutils literal"><span class="pre">date.fromordinal(d.toordinal())</span> <span class="pre">==</span> <span class="pre">d</span></tt>.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.weekday">
<tt class="descclassname">date.</tt><tt class="descname">weekday</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.weekday" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
For example, <tt class="docutils literal"><span class="pre">date(2002,</span> <span class="pre">12,</span> <span class="pre">4).weekday()</span> <span class="pre">==</span> <span class="pre">2</span></tt>, a Wednesday. See also
<a class="reference internal" href="#datetime.date.isoweekday" title="datetime.date.isoweekday"><tt class="xref py py-meth docutils literal"><span class="pre">isoweekday()</span></tt></a>.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.isoweekday">
<tt class="descclassname">date.</tt><tt class="descname">isoweekday</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.isoweekday" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the day of the week as an integer, where Monday is 1 and Sunday is 7.
For example, <tt class="docutils literal"><span class="pre">date(2002,</span> <span class="pre">12,</span> <span class="pre">4).isoweekday()</span> <span class="pre">==</span> <span class="pre">3</span></tt>, a Wednesday. See also
<a class="reference internal" href="#datetime.date.weekday" title="datetime.date.weekday"><tt class="xref py py-meth docutils literal"><span class="pre">weekday()</span></tt></a>, <a class="reference internal" href="#datetime.date.isocalendar" title="datetime.date.isocalendar"><tt class="xref py py-meth docutils literal"><span class="pre">isocalendar()</span></tt></a>.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.isocalendar">
<tt class="descclassname">date.</tt><tt class="descname">isocalendar</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.isocalendar" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a 3-tuple, (ISO year, ISO week number, ISO weekday).</p>
<p>The ISO calendar is a widely used variant of the Gregorian calendar. See
<a class="reference external" href="http://www.staff.science.uu.nl/~gent0113/calendar/isocalendar.htm">http://www.staff.science.uu.nl/~gent0113/calendar/isocalendar.htm</a> for a good
explanation.</p>
<p>The ISO year consists of 52 or 53 full weeks, and where a week starts on a
Monday and ends on a Sunday.  The first week of an ISO year is the first
(Gregorian) calendar week of a year containing a Thursday. This is called week
number 1, and the ISO year of that Thursday is the same as its Gregorian year.</p>
<p>For example, 2004 begins on a Thursday, so the first week of ISO year 2004
begins on Monday, 29 Dec 2003 and ends on Sunday, 4 Jan 2004, so that
<tt class="docutils literal"><span class="pre">date(2003,</span> <span class="pre">12,</span> <span class="pre">29).isocalendar()</span> <span class="pre">==</span> <span class="pre">(2004,</span> <span class="pre">1,</span> <span class="pre">1)</span></tt> and <tt class="docutils literal"><span class="pre">date(2004,</span> <span class="pre">1,</span>
<span class="pre">4).isocalendar()</span> <span class="pre">==</span> <span class="pre">(2004,</span> <span class="pre">1,</span> <span class="pre">7)</span></tt>.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.isoformat">
<tt class="descclassname">date.</tt><tt class="descname">isoformat</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.isoformat" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a string representing the date in ISO 8601 format, ‘YYYY-MM-DD’.  For
example, <tt class="docutils literal"><span class="pre">date(2002,</span> <span class="pre">12,</span> <span class="pre">4).isoformat()</span> <span class="pre">==</span> <span class="pre">'2002-12-04'</span></tt>.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.__str__">
<tt class="descclassname">date.</tt><tt class="descname">__str__</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.__str__" title="Permalink to this definition">¶</a></dt>
<dd><p>For a date <em>d</em>, <tt class="docutils literal"><span class="pre">str(d)</span></tt> is equivalent to <tt class="docutils literal"><span class="pre">d.isoformat()</span></tt>.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.ctime">
<tt class="descclassname">date.</tt><tt class="descname">ctime</tt><big>(</big><big>)</big><a class="headerlink" href="#datetime.date.ctime" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a string representing the date, for example <tt class="docutils literal"><span class="pre">date(2002,</span> <span class="pre">12,</span>
<span class="pre">4).ctime()</span> <span class="pre">==</span> <span class="pre">'Wed</span> <span class="pre">Dec</span> <span class="pre">4</span> <span class="pre">00:00:00</span> <span class="pre">2002'</span></tt>. <tt class="docutils literal"><span class="pre">d.ctime()</span></tt> is equivalent to
<tt class="docutils literal"><span class="pre">time.ctime(time.mktime(d.timetuple()))</span></tt> on platforms where the native C
<tt class="xref c c-func docutils literal"><span class="pre">ctime()</span></tt> function (which <a class="reference internal" href="time.html#time.ctime" title="time.ctime"><tt class="xref py py-func docutils literal"><span class="pre">time.ctime()</span></tt></a> invokes, but which
<a class="reference internal" href="#datetime.date.ctime" title="datetime.date.ctime"><tt class="xref py py-meth docutils literal"><span class="pre">date.ctime()</span></tt></a> does not invoke) conforms to the C standard.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.strftime">
<tt class="descclassname">date.</tt><tt class="descname">strftime</tt><big>(</big><em>format</em><big>)</big><a class="headerlink" href="#datetime.date.strftime" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a string representing the date, controlled by an explicit format string.
Format codes referring to hours, minutes or seconds will see 0 values. For a
complete list of formatting directives, see section
<a class="reference internal" href="#strftime-strptime-behavior"><em>strftime() and strptime() Behavior</em></a>.</p>
</dd></dl>

<dl class="method">
<dt id="datetime.date.__format__">
<tt class="descclassname">date.</tt><tt class="descname">__format__</tt><big>(</big><em>format</em><big>)</big><a class="headerlink" href="#datetime.date.__format__" title="Permalink to this definition">¶</a></dt>
<dd><p>Same as <a class="reference internal" href="#datetime.date.strftime" title="datetime.date.strftime"><tt class="xref py py-meth docutils literal"><span class="pre">date.strftime()</span></tt></a>. This makes it possible to specify format
string for a <a class="reference internal" href="#datetime.date" title="datetime.date"><tt class="xref py py-class docutils literal"><span class="pre">date</span></tt></a> object when using <a class="reference internal" href="stdtypes.html#str.format" title="str.format"><tt class="xref py py-meth docutils literal"><span class="pre">str.format()</span></tt></a>.
See section <a class="reference internal" href="#strftime-strptime-behavior"><em>strftime() and strptime() Behavior</em></a>.</p>
</dd></dl>

<p>Example of counting days to an event:</p>
<div class="highlight-python" style="position: relative;"><div class="highlight"><span class="copybutton" title="Hide the prompts and output" style="cursor: pointer; position: absolute; top: 0px; right: 0px; border: 1px solid rgb(170, 204, 153); color: rgb(170, 204, 153); font-family: monospace; padding-left: 0.2em; padding-right: 0.2em;">&gt;&gt;&gt;</span><pre>

<span class="gp">&gt;&gt;&gt; </span><span class="kn">import time</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from datetime import date</span>
</pre></div>
</div>



<p>Example of working with <a class="reference internal" href="#datetime.date" title="datetime.date"><tt class="xref py py-class docutils literal"><span class="pre">date</span></tt></a>:</p>
<div class="highlight-python" style="position: relative;"><div class="highlight"><span class="copybutton" title="Hide the prompts and output" style="cursor: pointer; position: absolute; top: 0px; right: 0px; border: 1px solid rgb(170, 204, 153); color: rgb(170, 204, 153); font-family: monospace; padding-left: 0.2em; padding-right: 0.2em;">&gt;&gt;&gt;</span><pre><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">datetime</span> <span class="kn">import</span> <span class="n">date</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">d</span> <span class="o">=</span> <span class="n">date</span><span class="o">.</span><span class="n">fromordinal</span><span class="p">(</span><span class="mi">730920</span><span class="p">)</span> <span class="c"># 730920th day after 1. 1. 0001</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">d</span>
<span class="go">datetime.date(2002, 3, 11)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">t</span> <span class="o">=</span> <span class="n">d</span><span class="o">.</span><span class="n">timetuple</span><span class="p">()</span>
<span class="gp">&gt;&gt;&gt; </span><span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">t</span><span class="p">:</span>
<span class="gp">... </span>    <span class="k">print</span> <span class="n">i</span>
<span class="go">2002                # year</span>
<span class="go">3                   # month</span>
<span class="go">11                  # day</span>
<span class="go">0</span>
<span class="go">0</span>
<span class="go">0</span>
<span class="go">0                   # weekday (0 = Monday)</span>
<span class="go">70                  # 70th day in the year</span>
<span class="go">-1</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">ic</span> <span class="o">=</span> <span class="n">d</span><span class="o">.</span><span class="n">isocalendar</span><span class="p">()</span>
<span class="gp">&gt;&gt;&gt; </span><span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">ic</span><span class="p">:</span>
<span class="gp">... </span>    <span class="k">print</span> <span class="n">i</span>
<span class="go">2002                # ISO year</span>
<span class="go">11                  # ISO week number</span>
<span class="go">1                   # ISO day number ( 1 = Monday )</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">d</span><span class="o">.</span><span class="n">isoformat</span><span class="p">()</span>
<span class="go">'2002-03-11'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">d</span><span class="o">.</span><span class="n">strftime</span><span class="p">(</span><span class="s">"</span><span class="si">%d</span><span class="s">/%m/%y"</span><span class="p">)</span>
<span class="go">'11/03/02'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">d</span><span class="o">.</span><span class="n">strftime</span><span class="p">(</span><span class="s">"%A </span><span class="si">%d</span><span class="s">. %B %Y"</span><span class="p">)</span>
<span class="go">'Monday 11. March 2002'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="s">'The {1} is {0:</span><span class="si">%d</span><span class="s">}, the {2} is {0:%B}.'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="s">"day"</span><span class="p">,</span> <span class="s">"month"</span><span class="p">)</span>
<span class="go">'The day is 11, the month is March.'</span>
</pre></div>
</div>
</div>
