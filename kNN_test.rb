require 'test/unit'
require_relative 'kNN'

class KnnTest < Test::Unit::TestCase

  def setup
    group = [[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]
    labels = ['A', 'A', 'B', 'B']
    @knn = Knn::Classifier.new(group, labels, 3)
  end
  
  
  def test_classify_0_0
    assert_equal "B", @knn.classify([0,0])
  end

  def test_classify_1_1
    assert_equal "A", @knn.classify([1, 1])
  end

  def test_classify_minus_1_1
    assert_equal "B", @knn.classify([-1, 1])
  end

  def test_classify_for_0_0_0
    group = [[0,0,0], [1, 0, 0], [1, 1, 1], [2, 2, 2], [7, 5, 2]]
    labels = ["A", "A", "B", "B", "C"]
    knn = Knn::Classifier.new(group, labels, 3)
    assert_equal "A", knn.classify([0.5, 0.5])
  end
    
end
