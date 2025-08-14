Meta: id=test/geometry/473.json, subject=Geometry, level=4
gt: 864 \mbox{ inches}^2  
pred: [ \boxed{864} \]
(high concencus)

Meta: id=test/precalculus/807.json, subject=Precalculus, level=2
gt: \left( 3, \frac{\pi}{2} \right)
pred: \boxed{(3, \frac{\pi}{2})} (turns into (3,\frac{\pi})
(high concencus)

Meta: id=test/precalculus/580.json, subject=Precalculus, level=3
gt: 120^\circ
pred: \boxed{120} (turns into 120)
(high concencus)

Meta: id=test/geometry/347.json, subject=Geometry, level=3
gt: 76^\circ
pred: \boxed{76} (turns into 76)
(high concencus)

Meta: id=test/precalculus/927.json, subject=Precalculus, level=4
same as above with 90

Meta: id=test/geometry/826.json, subject=Geometry, level=5
gt: 1\frac{4}{5}
(no concencus, all wrong) PRM also shows that

Meta: id=test/prealgebra/1139.json, subject=Prealgebra, level=5
gt: 4
pred: 3 (I know beam search can solve this)
(high concencus, all wrong)

Meta: id=test/intermediate_algebra/1197.json, subject=Intermediate Algebra, level=5
gt: \frac{3}{56}
pred: everyone here is wrong, beam search and BoN

Meta: id=test/intermediate_algebra/1388.json, subject=Intermediate Algebra, level=5
gt: 1,-2
pred: BoN and beam are actually correct with -2,1 (wrong comparison)

Meta: id=test/prealgebra/260.json, subject=Prealgebra, level=4
gt: 36^\circ
bon: 36
high concencus

Meta: id=test/precalculus/768.json, subject=Precalculus, level=5
gt: 3 \pm 2 \sqrt{2}
bon: \text{Nosolutions} (high)
beam: \text{Nosolutions} (high)

Meta: id=test/number_theory/488.json, subject=Number Theory, level=4
both correct but not super confident

Meta: id=test/prealgebra/1044.json, subject=Prealgebra, level=5
both confidently wrong

Meta: id=test/geometry/711.json, subject=Geometry, level=5
bon: confidently wrong
beam: sometimes correct
