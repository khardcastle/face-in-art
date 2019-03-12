$pdflatex = "texfot pdflatex -synctex=1 %O %S"; # maybe this should be moved to local config

$pdf_mode = 1;

push @generated_exts, 'nav';
push @generated_exts, 'snm';
push @generated_exts, 'tdo';
