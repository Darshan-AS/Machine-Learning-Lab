"""
@author: DON
"""
import numpy
from bokeh.io import show, output_file
from bokeh.plotting import figure


class Lowes:

    def __init__(self):
        self.X, self.y = None, None

    def fit(self, x_train, y_train):
        self.X = numpy.c_[numpy.ones(len(x_train)), x_train]
        self.y = y_train

    def predict(self, x_test, sigma):
        y_pred = numpy.array([])
        x_test = numpy.c_[numpy.ones(len(x_test)), x_test]

        for x0 in x_test:
            X_transpose_W = self.X.T * self.__gaussian_kernal(x0, sigma)
            beta = numpy.linalg.pinv(X_transpose_W @ self.X) @ X_transpose_W @ self.y
            y_pred = numpy.r_[y_pred, (x0 @ beta)]

        return y_pred

    def __gaussian_kernal(self, x0, sigma):
        return numpy.exp(numpy.sum((self.X - x0) ** 2, axis=1) / (-2 * sigma ** 2))


def main():
    x_train = numpy.linspace(0, 360, 130)
    y_train = numpy.sin(numpy.radians(x_train))
    noise = numpy.random.normal(loc=0, scale=.25, size=130)
    y_train += noise

    lowes = Lowes()
    lowes.fit(x_train, y_train)
    y_pred = lowes.predict(x_train, sigma=25)
    show_plot(x_train, y_train, y_pred)


def show_plot(x_train, y_train, y_pred):
    output_file('output.html', title='Locally Weighted Smoothing')
    plot = figure(title='Locally Weighted Smoothing',
                  x_axis_label='x (in degrees)',
                  y_axis_label='y = sin(x)')
    plot.scatter(x_train, y_train, legend='Actual dataset', color='green')
    plot.line(x_train, y_pred, legend='Predicted curve', color='red', line_width=2)
    show(plot)
    print('A file named "output.html" is downloaded in the current directory.\n'
          'Open it in the browser to see the plot.')


if __name__ == '__main__':
    main()
