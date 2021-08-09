import cv2
import numpy as np
import matplotlib
matplotlib.use('agg') # wip to fixup tkinter compatability issues
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def hough_line_detection(original_image, input_image, num_rhos=180, num_thetas=180, t_count=200):
  height, width = input_image.shape[:2]

  d = np.sqrt(np.square(height) + np.square(width))
  dtheta, drho = 180 / num_thetas, (2 * d) / num_rhos

  thetas = np.arange(0, 180, step=dtheta)
  rhos = np.arange(-d, d, step=drho)

  sin_thetas = np.sin(np.deg2rad(thetas))
  cos_thetas = np.cos(np.deg2rad(thetas))

  hough_lines = []
  inverse = False
  hough_matrix = np.zeros((len(rhos), len(rhos)))

  figure = plt.figure(figsize=(20, 7))
  plot1 = figure.add_subplot(1, 3, 1)
  plot1.imshow(original_image)
  plot2 = figure.add_subplot(1, 3, 2)
  plot2.imshow(original_image)
  plot3 = figure.add_subplot(1, 3, 3)
  plot3.imshow(original_image)

  for y in range(height):
    for x in range(width):
      if input_image[y][x] != 0:
        for theta_index in range(len(thetas)):
          rho = ((x - width / 2) * cos_thetas[theta_index]) + ((y - height / 2) * sin_thetas[theta_index])
          theta = thetas[theta_index]
          rho_index = np.argmin(np.abs(rhos - rho))
          hough_matrix[rho_index][theta_index] += 1

  for y in range(hough_matrix.shape[0]):
    for x in range(hough_matrix.shape[1]):
      if hough_matrix[y][x] > t_count:
        rho = rhos[y]
        theta = thetas[x]
        a, b = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))

        x0, y0 = (a * rho) + width / 2, (b * rho) + height / 2
        x1, y1 = int(x0 + 200 * (-b)), int(y0 + 200 * (a))
        x2, y2 = int(x0 - 200 * (-b)), int(y0 - 200 * (a))

        hough_lines.append({
          'strength': hough_matrix[y][x],
          'rho': rhos[y],
          'theta': thetas[x],
          'end_points': [x1, x2, y1, y2]
        })

        plot1.add_line(mlines.Line2D([x1, x2], [y1, y2]))

  # -- directionality determination
  if np.average([h["theta"] for h in hough_lines[:len(hough_lines)//2]]) < np.average([h["theta"] for h in hough_lines[len(hough_lines)//2:]]):
      inverse = True

  sort_fn = lambda x: (x.get('strength'))
  hough_lines = sorted(hough_lines, key=sort_fn, reverse=True)
  top_hough_lines = hough_lines[:5]

  for line in top_hough_lines:
      plot2.add_line(mlines.Line2D(line["end_points"][:2], line["end_points"][2:]))

  if inverse:
    for i in range(len(top_hough_lines)):
      top_hough_lines[i]["theta"] = 180 + top_hough_lines[i]["theta"]

  predicted_major_angle = np.mean([line["theta"] for line in top_hough_lines])
  m_x1, m_x2, m_y1, m_y2 = np.mean([hough_line["end_points"] for hough_line in top_hough_lines], axis=0)
  plot3.add_line(mlines.Line2D([m_x1, m_x2], [m_y1, m_y2], color=(0,1,0)))
  plot3.plot(m_x1,m_y1,'o',color="#00FF00") if inverse else plot2.plot(m_x2,m_y2,'o', color="#00FF00")

  plot1.title.set_text("Detected Hough Lines")
  plot2.title.set_text("Top 5 Hough Lines".format(predicted_major_angle))
  plot3.title.set_text("Predicted Major Hough Line ({}°)".format(predicted_major_angle))
  plt.savefig("images/hough_lines_test_output.png")

  return top_hough_lines, predicted_major_angle


if __name__ == "__main__":
  image = cv2.imread("images/hough_lines_test.png")
  image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_cvt = cv2.Canny(image_cvt, 100, 200)
  image_cvt = cv2.dilate(
      image_cvt,
      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
      iterations=1
  )
  image_cvt = cv2.erode(
      image_cvt,
      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
      iterations=1
  )
  a, b = hough_line_detection(image, image_cvt)
