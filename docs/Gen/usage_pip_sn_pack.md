

### Usage: pip_sn_pack.py [options] ###
<pre>
Options: 
  -h, --help            show this help message and exit 
  --package=PACKAGE
	package name to install [metrics]
--gitbranch=GITBRANCH
	gitbranch of the package [master] 
  --action=ACTION
	action to perform: list, install, uninstall [list]
</pre>

### Examples ###

<ul>
<li>  list installed packages
      <ul>
     <li> python pip_sn_pack.py --action list </li>
     </ul>
     </li>

 <li>  install packages (here sn_metrics)
       <ul>
       <li> python pip_sn_pack.py --action install --package sn_metrics </li>
       </ul>
       </li>
 <li>  uninstall packages (here sn_metrics)
       <ul>
       <li> python pip_sn_pack.py --action uninstall --package sn_metrics </li>
       </ul>
       </li>
</ul>
     