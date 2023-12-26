
save.files <- function(input.prop, input.weight, suffix){
  # save files -> makes sure to always save weight and otu matrix
  write.csv(t(input.weight),
            paste0("/Users/elisabeth.ailer/Projekte/P1_Microbiom/microbiom/CausalDiversity/generalIVmodel/Input/SIM_STATControl_weight", suffix, ".csv"))
  write.csv(t(sim.data.prop),
            paste0("/Users/elisabeth.ailer/Projekte/P1_Microbiom/microbiom/CausalDiversity/generalIVmodel/Input/SIM_STATControl_otuprop",  suffix, ".csv"))
}


compute.diversity <- function(otu){
  # compute shannon and simpson diversity
  otu.shannon <- diversity(otu, index="shannon")
  otu.simpson <- diversity(otu, index="simpson")
  return(list("shannon"=otu.shannon, "simpson"=otu.simpson))
}


compute.prop.otu <- function(otu){
  otu=otu[apply(otu,1, function(x) (sum(x>0)/length(x))>0.1),]
  ### remove taxaing which less than 10% individuals do not have.
  otu=otu[rowMeans(otu)>0.0001,]
  ### remove taxaing whose mean proportion less than e-4
  ######## filtering taxa which don't exist in control
  #otu=otu[rowSums(otu[,sam.1$Treatment=="control"])!=0,]
  otu.com=t(prop.table(otu+1,2))
  
  return(otu.com)
}
