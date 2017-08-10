open(FINA, "<", "INPUTPATH/allConsensualTerms_classes.txt") or die "cannot open";
open(FINB, "<", "INPUTPATH/allConsensualSpans_differentClasses.txt") or die "cannot open";
open(FOUT, ">", $ARGV[0]);

#read both files and create hash with class info for each word

##make hashes for consensual class anno
while(<FINA>)	{
	if ($_ =~ /^(.+)\t(.+)\t/)	{
		$term = $1;
		$val = $2;
	}
	else {print $_;} #output wrong lines to console
	if (!exists $data{$term}{$val})	{
		$data{$term}{$val} = 1;
	}
	else 	{$data{$term}{$val}++;}
}

##make hashes for conflicting class anno
while(<FINB>)	{
	if ($_ =~ /^(.+)\t(.+)\t(.+)$/)	{
		$term = $1;
		$val1 = $2;
		$val2 = $3;
	}
	else {print $_;} #output wrong lines to console
	if (!exists $data{$term}{$val1})	{
		$data{$term}{$val1} = 1;
	}
	else 	{$data{$term}{$val1}++;}
	if (!exists $data{$term}{$val2})	{
		$data{$term}{$val} = 1;
	}
	else 	{$data{$term}{$val2}++;}
}

@terms = keys %data;
#iterate through terms, finding the most frequent class
foreach $term (@terms)	{
	@classes = keys $data{$term};
	$highest = $data{$term}{$classes[0]}; #set first class to be highest initially
	push(@equals, $highest);
	$highest_class = $classes[0];
	for ($i = 1; $i <= $#classes; $i++)	{
		if ($data{$term}{$classes[$i]} > $highest)	{
			$highest = $data{$term}{$classes[$i]};
			$highest_class = $classes[$i];
		}
		elsif ($data{$term}{$classes[$i]} == $highest)	{	#deal with identical counts by random selection
			push(@equals, $i);#remember current index
		}
	}
	if ($#equals > 0)	{#at least one identical value has been found
		$randomised_highest = int(rand($#equals+1));
		$highest = $data{$term}{$classes[$randomised_highest]};
		$highest_class = $classes[$randomised_highest];
	}
	print FOUT $term, "\t", $highest_class, "\t", $highest, "\n";
	while (@equals)	{pop @equals;} #get rid of this array
}

close(FINA);
close(FINB);
close(FOUT);