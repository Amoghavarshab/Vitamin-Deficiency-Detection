-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Dec 13, 2024 at 08:33 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `vitamin_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `doctor`
--

CREATE TABLE `doctor` (
  `id` int(11) NOT NULL,
  `doctor_name` varchar(100) NOT NULL,
  `hospital_name` varchar(100) NOT NULL,
  `address` text NOT NULL,
  `location` varchar(100) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `doctor`
--

INSERT INTO `doctor` (`id`, `doctor_name`, `hospital_name`, `address`, `location`, `created_at`) VALUES
(1, 'Test', 'Vimala Hospital', '76-helo world', 'Koramangala', '2024-12-12 11:32:31'),
(2, 'Nelson', 'Apollo hospital', 'Electronic City, Bengaluru, Karnataka\r\n', 'Electronic City', '2024-12-12 12:03:55'),
(3, 'RAJEEV', 'MANIPAL', 'MANIPAL HOSPITAL HEBBAL', 'Hebbal', '2024-12-13 06:42:33');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `number` varchar(20) NOT NULL,
  `password` varchar(255) NOT NULL,
  `location` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `name`, `email`, `number`, `password`, `location`) VALUES
(1, 'Ajaycode', 'ajaycode@gmail.com', '9988776655', 'scrypt:32768:8:1$s82qlEDvScCmtogp$4a94c9fcb974ccd3bc84e3c896402a5c4b96eaed3527e305a1a8fb602ed9ebacacd03ce78a3c1fb8327726c5d9129deb9636c3268d7f433a21c3318789d30a30', 'Koramangala'),
(2, 'Test', 'test@gmail.com', '998877665544', 'scrypt:32768:8:1$RgxyKM4GR8Yhglbr$be9f43f482cdeea53645fe0e96eaeb3e269538bfd002863575757612cc2c796937ffc58407cd6984f6fd2a9404eb6b545b6c2d22493475493688cd600255ba4f', 'Electronic City'),
(3, 'damini', 'daminisriram@gmail.com', '9353150423', 'scrypt:32768:8:1$m57OnWVgTJtCkY9y$113d35bc989935c5ee0881ca2fe68e3cc1fe907e117b60edfb380dd1453436d526da25861d3d0c26555ed85d5dac0536a33964fed81747994b6f7d77e6b84ffa', 'Hebbal');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `doctor`
--
ALTER TABLE `doctor`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `doctor`
--
ALTER TABLE `doctor`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
