$do_cd = 1;

$out_dir = 'build';
$aux_dir = 'build';

$pdf_mode = 5;
$xelatex = 'xelatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

$bibtex = 'bibtex %O %B';
$bibtex_use = 2;

$clean_ext .= ' synctex.gz';
