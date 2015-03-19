
  (ns svm.core)
  (use 'clj-ml.data)

  (def raw-data (->> (->> "/home/murat/Clojure/Project/svm-path/dataset/negatives/"
                          java.io.File.
                          file-seq
                          (map #(vector % :negative))
                          rest)
                     (concat (->> "/home/murat/Clojure/Project/svm-path/dataset/positives/"
                                  java.io.File.
                                  file-seq
                                  (map #(vector % :positive))
                                  rest))))

  (defn read-image [[f t]]
    [(javax.imageio.ImageIO/read f) t])
  
  (defn get-pixel-color [img x y]
    (let [rgb (.getRGB img x y)
          color (java.awt.Color. rgb)] 
      (.getGreen color)))
  
  (def img-coords (for [x (range 0 40 10)
                        y (range 0 40 10)] [x y]))
  
  (defn average-section [img x y]
    (let [coords (->> (for [x (range x (+ x 10))
                            y (range y (+ y 10))] [x y])
                      (map (fn [[x y]] (get-pixel-color img x y))))]
      (int (/ (apply + coords) (count coords)))))
  
  (defn pixel-color-seq [img]
    (->> img-coords
         (reduce (fn[h [x y]] 
                   (conj h (average-section img x y)))
                 [])))
  
  (defn image-feature-map [img & [type]]
    (let [pxls (pixel-color-seq img)
          pxl (apply hash-map (interleave (map str (-> pxls count range)) pxls))]
      (if type
        (assoc pxl :kind type)
        pxl)))
  
  ;;(pixel-color-seq (first (read-image [(first (first raw-data)) nil])))
  
  (defn parse-dataset []
    (let [images (->> (map read-image raw-data)
                      (reduce (fn[h [img type]] 
                                (conj h (image-feature-map img type))) 
                              []))
          header (->> [(map str (-> images first count dec range))
                       {:kind [:positive :negative]}]
                      flatten
                      (into []))]
      [header images]))
  
  (defn dataset []
    (let [[header images] (parse-dataset)
          dataset (make-dataset "VisionSet" header (-> raw-data count))]
      (dataset-set-class dataset (-> header count dec))
      (doseq [img images]
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img))
        (.add dataset (make-instance dataset img)))
      dataset))

  (defn build-classifier [ds]
    (doto (weka.classifiers.functions.SMO.)
      (.buildClassifier ds)))
  
  (defn classify [classyfier ds instance]
    (.classifyInstance classyfier instance)
    (keyword (.value (.attribute ds (.classIndex ds)) (.classifyInstance classyfier instance))))
  
  (defn classify-image [dataset svm img]
    (let [image (-> [(java.io.File. img) nil] read-image first image-feature-map)]
      (classify svm dataset (make-instance dataset image))))
  

  ;  (def dataset (dataset))
  ;  (def vision-svm (build-classifier dataset))
  ;  (def test-data-negative (rest (file-seq (java.io.File. ""))))
   ; (def results-negative (map #(classify-image dataset vision-svm (str %)) test-data-negative))
  
   ; (classify-image dataset vision-svm "/home/murat/Clojure/Project/svm-path/dataset/negatives/2.png")
    

