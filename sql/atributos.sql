SELECT dteday AS data,
       LAG(cnt, 1) OVER (ORDER BY dteday) AS lag_1,
       LAG(cnt, 7) OVER (ORDER BY dteday) AS lag_7,
       AVG(cnt) OVER (
           ORDER BY dteday
           ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
       ) AS media_7
FROM demanda
ORDER BY dteday;
